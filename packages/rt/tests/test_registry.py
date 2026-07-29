"""The content door: keyed by (region, pair, threads, features); cuda refuses toward arrival."""

from dataclasses import dataclass

import pytest

from pdum.rt import Pair, acquire, cuda, executor_for, register_acquirer, register_compiler
from pdum.rt.select import Generator, Runtime


@dataclass(frozen=True)
class _FakeGen(Generator):
    pass


@dataclass(frozen=True)
class _FakeRt(Runtime):
    pass


class _Key:
    def __init__(self, key):
        self.key = key


class _Art:
    def __init__(self, key):
        self.region = _Key(key)


def test_cuda_refuses_toward_arrival():
    with pytest.raises(RuntimeError, match="arrives with the L4 team"):
        acquire(cuda.runtime)
    with pytest.raises(RuntimeError, match="arrives with the L4 team"):
        executor_for(_Art(("k",)), cuda)


def test_the_door_caches_and_never_collides():
    fake = Pair(_FakeGen(), _FakeRt())
    fake2 = Pair(Generator(), _FakeRt())
    register_acquirer(_FakeRt, lambda features: ("dev", features))
    made = []

    def compiler(art, device, threads):
        made.append((art.region.key, threads))
        return object()

    register_compiler(fake, compiler)
    register_compiler(fake2, compiler)

    art = _Art(("region", 1))
    a = executor_for(art, fake)
    assert executor_for(art, fake) is a  # same key -> the SAME executor
    assert executor_for(art, fake2) is not a  # a different pair NEVER collides
    assert executor_for(art, fake, thread_size=(8, 8, 1)) is not a  # threads SPECIALIZE
    assert len(made) == 3


def test_features_are_creation_time_and_key_devices():
    calls = []
    register_acquirer(_FakeRt, lambda features: calls.append(features) or ("dev", features))
    d1 = acquire(_FakeRt())
    d2 = acquire(_FakeRt(), features=("timestamps",))
    assert d1 != d2 and acquire(_FakeRt()) is d1
    assert calls == [(), ("timestamps",)] or calls[-1] == ("timestamps",)
