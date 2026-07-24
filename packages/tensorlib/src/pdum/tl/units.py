"""Exact units and quantities (step 3): the labeling algebra for charts.

Pint-inspired UX, exactness-first:

- Magnitudes are `fractions.Fraction` — never floats. Floats are rejected at
  every boundary because 0.1 is not 1/10, and exact membership ("is this
  coordinate on the lattice?") is the whole point of charts.
- A Unit is an exact rational scale factor onto base units plus a dimension
  exponent vector; conversion is exact by construction.
- Construct quantities from strings `q("0.75 um")`, rationals
  `Fraction(3, 4) * u.um`, ints `3 * u.mm`, or t-strings
  `q(t"{Fraction(3, 4)} um")` — PEP 750 interpolations carry ints/Fractions
  exactly, never through a float repr.

The unit system is deliberately minimal — a labeling algebra owned by this
library, not a physics package: no offset units (°C), no uncertainty, no
floats. `UnitRegistry.define` adds aliases (exact) or new base dimensions.
"""

from __future__ import annotations

import operator
import re
from dataclasses import dataclass
from fractions import Fraction
from string.templatelib import Template

# dimension exponent vector, e.g. (("length", 1), ("time", -1)) for velocity
Dims = tuple[tuple[str, int], ...]

_FLOAT_MSG = (
    "floats are inexact — write the value as a string ('0.1'), a Fraction, or a t-string interpolating an int/Fraction"
)


def _exact(value, what: str = "magnitude") -> Fraction:
    if isinstance(value, bool):
        raise TypeError(f"{what} {value!r}: bool is not a number here")
    if isinstance(value, float):
        raise TypeError(f"{what} {value!r}: {_FLOAT_MSG}")
    if isinstance(value, (int, Fraction)):
        return Fraction(value)
    if isinstance(value, str):
        try:
            return Fraction(value)  # handles "3/4", "0.75", "1e-3" exactly
        except ZeroDivisionError:
            raise ValueError(f"zero denominator in {what} {value!r}") from None
    try:
        return Fraction(operator.index(value))  # numpy ints and friends
    except TypeError:
        raise TypeError(f"cannot make an exact {what} from {value!r}") from None


def _merge_dims(a: Dims, b: Dims, sign: int) -> Dims:
    exp = dict(a)
    for name, e in b:
        exp[name] = exp.get(name, 0) + sign * e
        if exp[name] == 0:
            del exp[name]
    return tuple(sorted(exp.items()))


def _fmt_fraction(f: Fraction) -> str:
    """Prefer exact decimal display (0.25) over 1/4 when the denominator
    allows it; fall back to p/q."""
    if f.denominator == 1:
        return str(f.numerator)
    d = f.denominator
    two = five = 0
    while d % 2 == 0:
        d //= 2
        two += 1
    while d % 5 == 0:
        d //= 5
        five += 1
    if d == 1:
        k = max(two, five)
        scaled = abs(f.numerator) * 10**k // f.denominator
        digits = str(scaled).rjust(k + 1, "0")
        sign = "-" if f < 0 else ""
        return f"{sign}{digits[:-k]}.{digits[-k:]}"
    return f"{f.numerator}/{f.denominator}"


@dataclass(frozen=True, eq=False)
class Unit:
    scale: Fraction  # exact factor onto the base units of `dims`
    dims: Dims
    symbol: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "scale", Fraction(self.scale))
        if self.scale <= 0:
            raise ValueError("unit scale must be positive")

    # identity is (scale, dims); the symbol is cosmetic
    def __eq__(self, other):
        if not isinstance(other, Unit):
            return NotImplemented
        return self.scale == other.scale and self.dims == other.dims

    def __hash__(self):
        return hash((self.scale, self.dims))

    def __repr__(self) -> str:
        return self.symbol or "1"

    def __mul__(self, other):
        if isinstance(other, Unit):
            return Unit(
                self.scale * other.scale,
                _merge_dims(self.dims, other.dims, +1),
                f"{self.symbol}*{other.symbol}",
            )
        return Quantity(_exact(other), self)

    def __rmul__(self, other):  # 3 * u.mm
        return Quantity(_exact(other), self)

    def __truediv__(self, other):
        if isinstance(other, Unit):
            return Unit(
                self.scale / other.scale,
                _merge_dims(self.dims, other.dims, -1),
                f"{self.symbol}/{other.symbol}",
            )
        raise TypeError(f"cannot divide Unit by {other!r}")

    def __rtruediv__(self, other):  # 1 / u.s
        return Quantity(_exact(other), self**-1)

    def __pow__(self, n: int) -> "Unit":
        return Unit(
            self.scale**n,
            tuple((d, e * n) for d, e in self.dims if e * n != 0),
            f"{self.symbol}**{n}",
        )


ONE = Unit(Fraction(1), (), "")


@dataclass(frozen=True, eq=False)
class Quantity:
    magnitude: Fraction
    unit: Unit

    def __post_init__(self) -> None:
        object.__setattr__(self, "magnitude", _exact(self.magnitude))

    @property
    def base(self) -> Fraction:
        """Exact magnitude in base units."""
        return self.magnitude * self.unit.scale

    @property
    def dims(self) -> Dims:
        return self.unit.dims

    def to(self, unit) -> "Quantity":
        """Exact display-unit conversion."""
        if isinstance(unit, str):
            unit = u.parse_unit(unit)
        if unit.dims != self.dims:
            raise ValueError(f"cannot convert {self!r} to {unit!r}: dimensions differ")
        return Quantity(self.base / unit.scale, unit)

    # ---- arithmetic (exact; floats rejected by _exact) ----

    def _coerce(self, other) -> "Quantity | None":
        if isinstance(other, Quantity):
            return other
        if isinstance(other, (int, Fraction)):
            return Quantity(Fraction(other), ONE)
        return None

    def __add__(self, other):
        o = self._coerce(other)
        if o is None:
            return NotImplemented
        if o.dims != self.dims:
            raise ValueError(f"cannot add {self!r} and {other!r}: dimensions differ")
        return Quantity(self.magnitude + o.base / self.unit.scale, self.unit)

    __radd__ = __add__

    def __sub__(self, other):
        o = self._coerce(other)
        if o is None:
            return NotImplemented
        if o.dims != self.dims:
            raise ValueError(f"cannot subtract {other!r} from {self!r}: dimensions differ")
        return Quantity(self.magnitude - o.base / self.unit.scale, self.unit)

    def __rsub__(self, other):
        return -(self - other)

    def __neg__(self):
        return Quantity(-self.magnitude, self.unit)

    def __abs__(self):
        return Quantity(abs(self.magnitude), self.unit)

    def __bool__(self):
        return self.magnitude != 0

    def __mul__(self, other):
        if isinstance(other, Quantity):
            return Quantity(self.magnitude * other.magnitude, self.unit * other.unit)
        if isinstance(other, Unit):
            return Quantity(self.magnitude, self.unit * other)
        return Quantity(self.magnitude * _exact(other), self.unit)

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, Quantity):
            if other.dims == self.dims:
                return Fraction(self.base / other.base)  # exact dimensionless ratio
            return Quantity(self.magnitude / other.magnitude, self.unit / other.unit)
        return Quantity(self.magnitude / _exact(other), self.unit)

    def __rtruediv__(self, other):
        return Quantity(_exact(other) / self.magnitude, self.unit**-1)

    # ---- comparison: semantic (base magnitude + dims) ----

    def __eq__(self, other):
        o = self._coerce(other)
        if o is None:
            return NotImplemented
        return self.dims == o.dims and self.base == o.base

    def __hash__(self):
        if not self.dims:
            # dimensionless quantities equal plain numbers, so they must
            # hash like them (Python's __eq__/__hash__ contract)
            return hash(self.base)
        return hash((self.dims, self.base))

    def _cmp_base(self, other) -> tuple[Fraction, Fraction]:
        o = self._coerce(other)
        if o is None or o.dims != self.dims:
            raise ValueError(f"cannot compare {self!r} with {other!r}")
        return self.base, o.base

    def __lt__(self, other):
        a, b = self._cmp_base(other)
        return a < b

    def __le__(self, other):
        a, b = self._cmp_base(other)
        return a <= b

    def __gt__(self, other):
        a, b = self._cmp_base(other)
        return a > b

    def __ge__(self, other):
        a, b = self._cmp_base(other)
        return a >= b

    def __repr__(self) -> str:
        mag = _fmt_fraction(self.magnitude)
        return f"{mag} {self.unit.symbol}".strip()


def _template_to_str(t: Template) -> str:
    """Assemble a t-string exactly: int/Fraction interpolations via their
    exact text form; floats rejected."""
    parts = []
    interps = t.interpolations
    for i, s in enumerate(t.strings):
        parts.append(s)
        if i < len(interps):
            v = interps[i].value
            if isinstance(v, float):
                raise TypeError(f"t-string interpolation {v!r}: {_FLOAT_MSG}")
            if isinstance(v, (int, Fraction, str)):
                parts.append(str(v))
            else:
                raise TypeError(f"t-string interpolation {v!r}: expected int/Fraction/str")
    return "".join(parts)


_MAGNITUDE = re.compile(r"\s*([+-]?(?:\d+/\d+|(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?))\s*(.*?)\s*$")
_ATOM = re.compile(r"(1|[A-Za-z_]+)(?:\^(-?\d+))?$")


class UnitRegistry:
    """A pint-flavored registry with exact rational conversions.

    `reg.um` / `reg["um"]` look up units; `reg.define` adds an alias with an
    exact scale (`define("min", "60 s")`) or a brand-new base dimension
    (`define("px", dim="pixel")`); `reg.quantity` parses strings/t-strings.
    """

    def __init__(self, seed: bool = True):
        self._units: dict[str, Unit] = {}
        if seed:
            self._seed()

    def define(self, name: str, reference=None, *, dim: str | None = None) -> Unit:
        if (reference is None) == (dim is None):
            raise ValueError("give exactly one of: reference quantity, dim=")
        if dim is not None:
            unit = Unit(Fraction(1), ((dim, 1),), name)
        else:
            qty = self.quantity(reference) if isinstance(reference, (str, Template)) else reference
            if not isinstance(qty, Quantity):
                raise TypeError(f"reference must be a Quantity or parseable string, got {reference!r}")
            unit = Unit(qty.base, qty.dims, name)
        self._units[name] = unit
        return unit

    def __getattr__(self, name: str) -> Unit:
        try:
            return self._units[name]
        except KeyError:
            raise AttributeError(f"no unit {name!r} defined") from None

    def __getitem__(self, name: str) -> Unit:
        return self._units[name]

    def parse_unit(self, text: str) -> Unit:
        """Parse a unit expression. `*` and `/` associate LEFT TO RIGHT with
        equal precedence (pint's convention): m/s*s == (m/s)*s == m. `**` (or
        `^`) binds an integer exponent to a single symbol."""
        text = text.strip()
        if not text or text == "1":
            return ONE

        def atom_unit(atom: str) -> Unit:
            m = _ATOM.fullmatch(atom)
            if m is None:
                raise ValueError(f"cannot parse unit {atom!r} in {text!r}")
            name, exp = m.group(1), int(m.group(2) or 1)
            base = ONE if name == "1" else self[name]
            return base**exp

        tokens = [tk.strip() for tk in re.split(r"([*/])", text.replace("**", "^"))]
        if not tokens[0]:
            raise ValueError(f"cannot parse unit {text!r}")
        unit = atom_unit(tokens[0])
        for i in range(1, len(tokens), 2):
            op = tokens[i]
            atom = tokens[i + 1] if i + 1 < len(tokens) else ""
            if not atom:
                raise ValueError(f"dangling {op!r} in unit {text!r}")
            unit = unit * atom_unit(atom) if op == "*" else unit / atom_unit(atom)
        return Unit(unit.scale, unit.dims, text)

    def quantity(self, spec) -> Quantity:
        if isinstance(spec, Quantity):
            return spec
        if isinstance(spec, (int, Fraction)):
            return Quantity(Fraction(spec), ONE)
        if isinstance(spec, Template):
            spec = _template_to_str(spec)
        if isinstance(spec, float):
            raise TypeError(f"{spec!r}: {_FLOAT_MSG}")
        if not isinstance(spec, str):
            raise TypeError(f"cannot parse a quantity from {spec!r}")
        m = _MAGNITUDE.fullmatch(spec)
        if m is None:
            raise ValueError(f"cannot parse quantity {spec!r}")
        try:
            magnitude = Fraction(m.group(1))
        except ZeroDivisionError:
            raise ValueError(f"zero denominator in quantity {spec!r}") from None
        unit = self.parse_unit(m.group(2)) if m.group(2) else ONE
        return Quantity(magnitude, unit)

    def _seed(self) -> None:
        d = self.define
        d("m", dim="length")
        d("km", "1000 m")
        d("cm", "0.01 m")
        d("mm", "0.001 m")
        d("um", "1e-6 m")
        d("nm", "1e-9 m")
        d("s", dim="time")
        d("ms", "0.001 s")
        d("us", "1e-6 s")
        d("ns", "1e-9 s")
        d("min", "60 s")
        d("h", "3600 s")
        d("kg", dim="mass")
        d("g", "0.001 kg")
        d("mg", "1e-6 kg")
        d("Hz", "1 s**-1")


u = UnitRegistry()
q = u.quantity
