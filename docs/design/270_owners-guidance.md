# 270 — Owner's Guidance (the efficiency era)

Dictated by the owner at the P9 close, for the work starting off
`main`. This rides ON TOP of `260_l4-brief.md` (the mechanical
onboarding) and wins where they differ. Read 260 for what is ruled and
frozen; read this for what the work IS.

## The arc

The work to date was FOUNDATIONS — the caching mechanism; lowering in
the dsl layer and in tensorlib; layouts, buffers, and the primitive
operations for both layout and compute — plus EXPRESSIVITY: proving
the substrate covers the use cases we care about, from deep learning
to physics/PDE solving to graphics pipelines. The primary focus now
shifts: from an expressive framework to an EFFICIENT one —
infrastructure that takes the programs already written and transforms
them into efficient code for specific hardware. Other kinds of work
continue alongside; the literal first task is removing the superseded
legacy IR (the excavation, LEVELS.md).

## The core thesis: no magic compile function

We are NOT building a compiler that takes a denotational spec plus a
machine description and magically produces an efficient program.
Instead: a program undergoes a sequence of transformations, and **at
every point in the process it remains a runnable program on a
backend** — at some point in the chain it is something that can be
COMPILED to a backend, and passes may keep going after that. You
transform, run, measure, transform again. This is incremental
compilation only in the sense that **we are constantly measuring as we
compile** — never the traditional sense. The machine description
exists (the hardware tree: memory and compute as the major elements)
and informs the transformations; it never feeds a magic function.

## The aspects the passes target

- **Lower precision.**
- **Kernel fusion** — deciding which pieces of a linear program fuse
  into one GPU kernel, where a kernel represents one host→device
  invocation. This is load-bearing: our users have only a few
  computational primitives, where other frameworks get GPU boundaries
  implicitly baked into neural modules. If the efficient program is
  not RECOGNIZED here, the result is badly inefficient in memory and
  everything else.
- **Buffer reuse** for intermediates, so the program is not constantly
  allocating/deallocating. Honest caveat, recorded: modern GPU
  allocation may be simple enough that this matters less than
  expected — an open question, but the capability is still worth
  building.
- **Activation checkpointing** for programs that do not fit in memory.
- **Distributed mapping** across GPUs on a node and across nodes. Key
  insight: the transformation PRIMITIVES here strongly resemble the
  tile-level ones — the choices differ (compute-vs-bandwidth
  trade-offs), but the same transformer vocabulary can serve the
  different stages of the hierarchy.
- **New expressivity**: a tiling language and a warp-intrinsics
  language — a layer not yet designed, to be integrated with the rest.
  Open question: once fusion boundaries are decided, is a given kernel
  expressed in the tiling language or in the compute-kernel language?
  Some things may be simpler one way or the other.
- **AD interaction**: how the forward pass is broken into kernels need
  not match how the backward pass fuses — a genuine benefit of the
  whole-program approach; do not artificially couple them.

The primary input throughout is an assemblage program.

## The workflow: rational optimization, incremental investment

Build ANALYSES that make rational optimization possible; do not apply
every optimization at once. Take a program whose empirical value is
unproven, run it on a small dataset, analyze, and invest more
compilation effort only as it proves promising — scaling the
architecture and the compilation investment together. Initially,
"compiling" is an AGENT handed a function and five minutes to try
transformations with the toolkit and measure. A concrete compiler
function for a large domain may eventually emerge from this — that is
the endpoint, never the premise.

## Waves, and backend urgency

Do not drift far from a concrete backend — get to CUDA quickly. Work
in waves: a first wave puts foundational pieces in place covering A
LITTLE OF EVERY ASPECT, so everything is at least reasonably
represented and running on real hardware; later waves progressively
deepen. Each aspect (bufferization is the standing example) has many
levels of sophistication to choose among — choosing the level per wave
is itself a decision. The waves need not be strictly sequential. The
anti-pattern we have genuinely suffered from and must avoid: baking in
premature features that later have to be relaxed or undone.

## Host machinery and tooling

- Host-level tools to load and unload things between host and device.
- Streams and events; profiling as a first-class capability.
- Propagate SOURCE LOCATIONS through lowering — keep them if we can.
- Probably no PTX (imaginable, but not the plan).
- Integrate nsys and the compute/systems profiler tools deeply enough
  that, for code compiled all the way down to machine level, the
  framework itself can answer questions like registers-per-kernel, and
  profiling information is captured easily.

## The standing constraint, restated for this era

No magic compile function. No automatic casts or conversions. No
convenience affordances. This code is written mostly BY AI, and the
hypothesis is that the AI should prefer well-structured, semantically
clear code — it does not mind an explicit cast that is not strictly
necessary, or explicit dimensions where a human would want implicit
broadcast. We do not give it those affordances. Explicit, always.
