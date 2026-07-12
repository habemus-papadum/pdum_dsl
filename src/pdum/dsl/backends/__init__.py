"""Backends are SATELLITES: each registers itself into the DEFAULT registry
at import, with zero kernel edits. Importing this package wires all bundled
backends (today: python)."""

from . import python  # noqa: F401
