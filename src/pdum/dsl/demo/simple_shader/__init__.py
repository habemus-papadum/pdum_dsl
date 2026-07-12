"""The simple-shader demo pair: scalar kernels, staging-as-uniforms, the
compute-family v1 contract (params are thread coordinates). Registration
names are dotted cell names: ``demo.simple_shader.python`` (default, kind
"device") and ``demo.simple_shader.wgsl.{compute,fragment}`` (routed)."""

from . import python, wgsl  # noqa: F401
