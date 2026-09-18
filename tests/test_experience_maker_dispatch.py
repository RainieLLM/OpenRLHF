"""Dispatch/barrier ordering contract for RemoteExperienceMaker.make_experience.

The four model forwards are dispatched to Ray actor groups whose GPUs are shared
depending on the --train.colocate_* flags:

* ``colocate_all``            -- every model shares one set of GPUs.
* ``colocate_actor_ref``      -- actor and reference share GPUs.
* ``colocate_critic_reward``  -- critic and reward share GPUs.

A model must not be dispatched while a co-located model still occupies the same
GPUs, so each pair needs a barrier. A barrier that guards a *later* dispatch must
not be placed in front of an unrelated dispatch, otherwise the driver serialises
work that could have overlapped.

These tests pin the dispatch order and the barrier placement for each flag
combination without needing a GPU, a Ray cluster, or a model checkpoint.
"""

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def _install_module(name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


_STUBBED_MODULES = (
    "ray",
    "openrlhf",
    "openrlhf.models",
    "openrlhf.models.utils",
    "openrlhf.trainer",
    "openrlhf.trainer.ppo_utils",
    "openrlhf.trainer.ppo_utils.experience",
    "openrlhf.trainer.ppo_utils.length_penalty",
    "openrlhf.trainer.ray",
    "openrlhf.trainer.ray.launcher",
    "openrlhf.utils",
    "openrlhf.utils.logging_utils",
    "openrlhf.utils.seqlen_balancing",
)

_MISSING = object()


def _load_experience_maker(events):
    """Load experience_maker.py with a recording stand-in for ray.

    The stubs are installed only for the duration of the import and then restored,
    so this module cannot leak stand-in packages into the rest of the test session.
    """
    root = Path(__file__).resolve().parents[1]
    saved = {name: sys.modules.get(name, _MISSING) for name in _STUBBED_MODULES}

    try:
        for package in (
            "openrlhf",
            "openrlhf.models",
            "openrlhf.trainer",
            "openrlhf.trainer.ppo_utils",
            "openrlhf.trainer.ray",
            "openrlhf.utils",
        ):
            package_module = types.ModuleType(package)
            package_module.__path__ = []
            sys.modules[package] = package_module

        ray_module = types.ModuleType("ray")

        def _ray_get(ref):
            events.append(f"barrier:{ref}")
            return [[None]]

        ray_module.get = _ray_get
        ray_module.put = lambda obj: "dummy"
        sys.modules["ray"] = ray_module

        _install_module(
            "openrlhf.models.utils",
            compute_approx_kl=lambda *a, **k: torch.zeros(1),
            compute_reward=lambda *a, **k: torch.zeros(1),
            masked_mean=lambda *a, **k: torch.zeros(1),
        )
        _install_module("openrlhf.trainer.ppo_utils.experience", Experience=object)
        _install_module("openrlhf.trainer.ppo_utils.length_penalty", apply_length_penalties=lambda *a, **k: None)
        _install_module("openrlhf.trainer.ray.launcher", RayActorGroup=object)
        _install_module("openrlhf.utils.logging_utils", init_logger=lambda name: SimpleNamespace(info=lambda *a: None))
        _install_module(
            "openrlhf.utils.seqlen_balancing",
            get_minimum_num_micro_batch_size=lambda *a, **k: 1,
            get_seqlen_balanced_partitions=lambda *a, **k: [[0]],
        )

        spec = importlib.util.spec_from_file_location(
            "_openrlhf_experience_maker_dispatch_test",
            root / "openrlhf" / "trainer" / "ppo_utils" / "experience_maker.py",
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        for name, original in saved.items():
            if original is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


class _RecordingGroup:
    """Stands in for RayActorGroup, recording dispatch and empty_cache calls."""

    def __init__(self, name, events):
        self.name = name
        self.events = events

    def async_run_method_batch(self, method_name, **kwargs):
        self.events.append(f"dispatch:{self.name}")
        return f"{self.name}.{method_name}"

    def async_run_method(self, method_name, **kwargs):
        self.events.append(f"{method_name}:{self.name}")
        return f"{self.name}.{method_name}"


def _make_args(colocate_all=False, colocate_actor_ref=False, colocate_critic_reward=False):
    return SimpleNamespace(
        train=SimpleNamespace(
            colocate_all=colocate_all,
            colocate_actor_ref=colocate_actor_ref,
            colocate_critic_reward=colocate_critic_reward,
        ),
        ds=SimpleNamespace(ring_attn_size=1, tensor_parallel_size=1),
        algo=SimpleNamespace(
            kl=SimpleNamespace(use_loss=False, estimator="k3"),
            advantage=SimpleNamespace(estimator="group_norm"),
        ),
        rollout=SimpleNamespace(n_samples_per_prompt=1, micro_batch_size=1),
        reward=SimpleNamespace(clip_range=0.0),
    )


def _record_dispatch_events(with_critic=True, with_reference=True, **flags):
    events = []
    module = _load_experience_maker(events)

    maker = module.RemoteExperienceMaker(
        actor_model_group=_RecordingGroup("actor", events),
        critic_model_group=_RecordingGroup("critic", events) if with_critic else None,
        reward_model_group=_RecordingGroup("reward", events),
        initial_model_group=_RecordingGroup("reference", events) if with_reference else None,
        kl_controller=SimpleNamespace(value=0.0),
        strategy=SimpleNamespace(args=_make_args(**flags)),
        tokenizer=SimpleNamespace(pad_token_id=0),
    )
    maker._flatten_results = lambda refs, duplicate_factor: [torch.zeros(1)]

    sample = SimpleNamespace(
        rewards=None,
        action_mask=torch.zeros(1),
        sequences=torch.zeros(1),
        attention_mask=torch.zeros(1),
        mm_train_inputs=None,
        info={},
    )
    maker.make_experience([sample])
    return events


def _index(events, needle):
    return events.index(needle)


def test_dispatches_every_model_forward_once():
    events = _record_dispatch_events(colocate_actor_ref=True)
    dispatched = [event for event in events if event.startswith("dispatch:")]
    assert dispatched == ["dispatch:reward", "dispatch:actor", "dispatch:critic", "dispatch:reference"]


def test_colocate_actor_ref_does_not_delay_critic_dispatch():
    """Critic shares no GPUs with actor here, so it must be dispatched before the actor barrier."""
    events = _record_dispatch_events(colocate_actor_ref=True)

    assert _index(events, "dispatch:critic") < _index(events, "barrier:actor.forward")
    assert _index(events, "barrier:actor.forward") < _index(events, "dispatch:reference")


def test_colocate_critic_reward_does_not_delay_reference_dispatch():
    """Reference shares no GPUs with critic here, so it must be dispatched before the critic barrier."""
    events = _record_dispatch_events(colocate_critic_reward=True)

    assert _index(events, "dispatch:reference") < _index(events, "barrier:critic.forward")
    assert _index(events, "barrier:reward.forward") < _index(events, "dispatch:critic")


def test_both_flags_overlap_critic_with_actor_and_reference_with_critic():
    events = _record_dispatch_events(colocate_actor_ref=True, colocate_critic_reward=True)

    assert _index(events, "dispatch:critic") < _index(events, "barrier:actor.forward")
    assert _index(events, "dispatch:reference") < _index(events, "barrier:critic.forward")


def test_colocate_all_keeps_every_forward_strictly_sequential():
    """With every model on one set of GPUs each forward must be released before the next dispatch."""
    events = _record_dispatch_events(colocate_all=True)

    for earlier, later in (
        ("reward", "actor"),
        ("actor", "critic"),
        ("critic", "reference"),
    ):
        assert _index(events, f"empty_cache:{earlier}") < _index(events, f"dispatch:{later}")


@pytest.mark.parametrize("colocate_actor_ref", [False, True])
def test_reference_cache_is_released_only_when_it_shares_gpus(colocate_actor_ref):
    """The reference forward frees its cache for the training step only if it shares GPUs."""
    events = _record_dispatch_events(colocate_actor_ref=colocate_actor_ref)

    if colocate_actor_ref:
        assert _index(events, "dispatch:reference") < _index(events, "empty_cache:reference")
    else:
        assert "empty_cache:reference" not in events


def test_colocate_all_overrides_the_individual_flags():
    """--colocate_all puts every model on one GPU set, so the individual flags change nothing."""
    sequential = _record_dispatch_events(colocate_all=True)
    sequential_with_flags = _record_dispatch_events(
        colocate_all=True, colocate_actor_ref=True, colocate_critic_reward=True
    )

    assert sequential == sequential_with_flags


@pytest.mark.parametrize("with_critic", [False, True])
@pytest.mark.parametrize("with_reference", [False, True])
def test_missing_model_groups_do_not_break_the_dispatch_sequence(with_critic, with_reference):
    """A run without a critic or a reference model still dispatches the remaining forwards."""
    events = _record_dispatch_events(
        with_critic=with_critic,
        with_reference=with_reference,
        colocate_actor_ref=True,
        colocate_critic_reward=True,
    )

    dispatched = [event for event in events if event.startswith("dispatch:")]
    expected = ["dispatch:reward", "dispatch:actor"]
    if with_critic:
        expected.append("dispatch:critic")
    if with_reference:
        expected.append("dispatch:reference")

    assert dispatched == expected
    assert "barrier:dummy" not in events
