"""The registry — device acquisition and the executor content door.

Two responsibilities, both pattern-setting for every column (283 §3):

ACQUISITION. ``acquire(runtime, features=())`` returns the live device
for a Runtime, cached per (runtime, features). Features are requested
AT CREATION — the lesson of the old ``graphics._device()`` singleton,
which took no feature requests and made timestamp queries unreachable
without monkeypatching. Columns register an acquirer; a runtime with
no registered acquirer refuses naming its arrival.

THE CONTENT DOOR. ``executor_for(art, pair, ...)`` compiles a device
executor for an artifact THROUGH a content-keyed cache — never around
it (the FAIL-5 lesson: the old ``wgpu_artifact`` bypassed the door via
``dataclasses.replace``, so its declared fp keyed nothing and every
swap recompiled). The key is
``(region key, pair fp, thread size, features)``: the pair enters
because two columns' artifacts share a region key and must never share
a row; thread size enters because it SPECIALIZES (owner-ruled). The
compiled executor has the launch protocol's shape —
``(values, staging) -> effect`` — so the kernel tier's launcher runs
staging/rebind/overlap identically and lands on the device.

rt never imports pdum.tl: artifacts arrive duck-typed (``region``,
``tensor_params``, ``writable``, ``uniforms``, ``arg_slots``) and
regions are walked structurally. That direction is what lets the
tensor tier import rt without a cycle.
"""

from __future__ import annotations

from pdum.dsl.cache import Memo

from .select import Pair, Runtime

_ACQUIRERS: dict[type, object] = {}  # Runtime type -> fn(features) -> device
_DEVICES: dict[tuple, object] = {}  # (runtime fp, features) -> live device
_COMPILERS: dict[Pair, object] = {}  # pair -> fn(art, device, thread_size) -> executor

EXECUTORS = Memo("rt.executor", capacity=1 << 20)


def register_acquirer(runtime_type: type, fn) -> None:
    _ACQUIRERS[runtime_type] = fn


def register_compiler(pair: Pair, fn) -> None:
    _COMPILERS[pair] = fn


def acquire(runtime: Runtime, features: tuple[str, ...] = ()):
    """The device, acquired once per (runtime, features) — features are
    requested at creation because they cannot be added later."""
    fn = _ACQUIRERS.get(type(runtime))
    if fn is None:
        raise RuntimeError(
            f"no acquirer is registered for {type(runtime).__name__} — the cuda column "
            f"arrives with the L4 team (283 §5); webgpu/metal register on import of "
            f"pdum.rt"
        )
    key = (runtime.fp(), tuple(sorted(features)))
    if key not in _DEVICES:
        _DEVICES[key] = fn(tuple(sorted(features)))
    return _DEVICES[key]


def executor_for(art, pair: Pair, *, thread_size: tuple[int, ...] | None = None, features: tuple[str, ...] = ()):
    """A compiled device executor for ``art`` on ``pair`` — through the
    content door. Same artifact + same pair = the same executor object;
    a different pair never collides (region keys are shared across
    columns by construction)."""
    fn = _COMPILERS.get(pair)
    if fn is None:
        raise RuntimeError(
            f"no compiler is registered for {pair} — the cuda column arrives with the "
            f"L4 team (283 §5); webgpu/metal register on import of pdum.rt"
        )
    key = (art.region.key, pair.fp(), ("threads", thread_size), tuple(sorted(features)))
    return EXECUTORS.get_or_compile(key, lambda: fn(art, acquire(pair.runtime, features), thread_size))
