"""The scope (200 §1.6/1.7): your position in the model's one address space.

An explicit, immutable value carrying that path's facets — path, the shared
collection, the randomness root, a small policy map — and interpreting NONE
of them. Thin by law. The flat name space is primary: ``s / "h" / "3"`` is a
prefix view, and ``s.param("wq", ...)`` declares the leaf ``h.3.wq``.

- **param**: declares a leaf (name, dims, extents; NO buffer) and returns a
  virtual tensor the makers capture. Declaration is idempotent; a conflict
  refuses — contract names are never auto-suffixed.
- **randomness**: streams derive from site paths (``fold_in(root,
  path)``) — no key threads through model code. (The root rides the scope
  as a build-time key at the reference tier; the program-input form arrives
  with @compute.)
- **taps**: ``tap(x, s / "k")`` marks a potential output under its site
  path and returns x unchanged; unrequested taps are pruned and cost
  nothing.
- **policies**: an open string-keyed set (``s.with_(mode="eval")``) the
  scope never interprets — and they are IDENTITY-BEARING: the policy map
  folds into build identity, so a train build and an eval build can never
  collide in the cache.

There is no module-level "current scope" stack, ever. The only mutation
anywhere is the collection's build-time registration.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from pdum.dsl.naming import NameCollision

from .ir import _dense_like
from .layout import Dim
from .lifting import _Intrinsic

tap = _Intrinsic("tap")
dropout = _Intrinsic("dropout")


@dataclass(frozen=True)
class Param:
    """A declared leaf: a VIRTUAL tensor — layout and carrier, no buffer.
    Makers capture these; capture identity decides leaf identity (the tie:
    one object captured twice is ONE input leaf)."""

    name: str  # the flat contract name, e.g. "h.3.attn.wq"
    dims: tuple  # ((dim, extent), ...) in declaration order

    @property
    def layout(self):
        return _dense_like(tuple(Dim(n, 0, 0, e) for n, e in self.dims))


@dataclass
class Collection:
    """The one mutable thing: build-time registration of leaves and taps."""

    leaves: dict = field(default_factory=dict)  # name -> Param


@dataclass(frozen=True)
class Scope:
    path: tuple = ()
    coll: Collection = field(default_factory=Collection)
    policies: tuple = ()  # sorted (key, value) items — hashable, identity-bearing
    root_key: int = 0

    def __truediv__(self, name: str) -> "Scope":
        return replace(self, path=self.path + (str(name),))

    @property
    def name(self) -> str:
        return ".".join(self.path)

    def param(self, name: str, **dims: int) -> Param:
        full = ".".join(self.path + (name,))
        spec = tuple(dims.items())
        existing = self.coll.leaves.get(full)
        if existing is not None:
            if existing.dims == spec:
                return existing  # idempotent: the SAME object (capture identity)
            raise NameCollision(
                f"leaf {full!r} is already declared with dims {dict(existing.dims)} — "
                f"contract names are never auto-suffixed; declare it once, or "
                f"address a different path"
            )
        p = Param(full, spec)
        self.coll.leaves[full] = p
        return p

    def with_(self, **policies) -> "Scope":
        merged = dict(self.policies) | policies
        return replace(self, policies=tuple(sorted(merged.items())))

    def policy(self, key: str, default=None):
        return dict(self.policies).get(key, default)

    def stream(self, *suffix: str) -> int:
        """The site's randomness stream: fold_in(root, path) — name-derived,
        insertion-stable, refactor-stable."""
        from .random import fold_in

        return fold_in(self.root_key, ".".join(self.path + suffix) or "<root>")

    def seq(self, name: str, maker, cfg, n: int):
        """The one sequencing combinator — the explicit host loop, named,
        thin enough to print (§6.3)."""
        from .assemblage import pipe

        units = [maker(self / name / str(i), cfg(i) if callable(cfg) else cfg) for i in range(n)]
        return pipe(units)

    def spec(self) -> dict:
        """The derived parameter table: name -> dims. Code is authoritative;
        this is a VIEW of what the makers declared."""
        return {name: dict(p.dims) for name, p in sorted(self.coll.leaves.items())}


def scope(root_key: int = 0) -> Scope:
    """A fresh root scope: empty path, empty collection, no policies."""
    return Scope(root_key=root_key)
