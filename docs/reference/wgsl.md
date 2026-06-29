# API Reference — WGSL Backend

Emitting WGSL text and the uniform-buffer layout. This backend consumes inlined, typed IR
and produces a [`WgslModule`](#pdum.dsl.backends.wgsl.compile); the
[runtime](runtime.md) turns that into a real pipeline.

## Compile entry — `pdum.dsl.backends.wgsl.compile`

::: pdum.dsl.backends.wgsl.compile

## Uniform layout — `pdum.dsl.backends.wgsl.layout`

::: pdum.dsl.backends.wgsl.layout

## Emission — `pdum.dsl.backends.wgsl.emit`

::: pdum.dsl.backends.wgsl.emit

## Intrinsics & dialect tables — `pdum.dsl.backends.wgsl.intrinsics`

::: pdum.dsl.backends.wgsl.intrinsics
