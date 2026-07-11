"""WGSL backend: typed IR → WGSL text + uniform-buffer layout."""

from .compile import WgslModule, compile_fragment

__all__ = ["WgslModule", "compile_fragment"]
