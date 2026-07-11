# R3 — xDSL and the MLIR concept set: candidate answer to the IR question

Research report for the pdum.dsl redesign. Investigated July 11, 2026, against live
sources: the xDSL GitHub repo (cloned at commit `175b9ec`, 2026-07-09), PyPI
release 0.68.0, the readthedocs site, and the CGO 2025 paper. All API claims below
were **executed against xdsl 0.68.0 on Python 3.14** (venv install), not read from
docs — the working exercise script is reproduced in §3.6.

Sources:
- Repo: https://github.com/xdslproject/xdsl
- PyPI: https://pypi.org/project/xdsl/ (v0.68.0, released 2026-07-03)
- Docs: https://xdsl.readthedocs.io/latest/ and https://xdsl.dev/
- Releases: https://github.com/xdslproject/xdsl/releases
- Paper: Fehr et al., *xDSL: Sidekick Compilation for SSA-Based Compilers*, CGO 2025,
  https://steuwer.info/files/publications/2025/CGO-xDSL.pdf
  (DOI 10.1145/3696443.3708945)
- Ecosystem: https://xdsl.dev/news/, https://xdsl.dev/events/2025-HiPEAC/,
  https://maintainermonth.github.com/academia/xdsl-maintainer-spotlight

---

## 1. Executive verdict

xDSL is real, alive, well-engineered, and its core concept set (dialects, IRDL
declarative op definitions, regions, traits, the value/attribute split, worklist
pattern rewriting) is almost eerily well-matched to pdum.dsl's needs — I defined a
`pdum.uniform` op with a compile-time slot **property** and a runtime SSA
**result**, verified it, round-tripped it through the textual format, rewrote it,
and interpreted it against a mutable runtime env in ~70 lines total. That is
precisely the uniform-vs-Literal distinction, expressed natively.

But: it is a 0.x project with **deliberate, frequent breaking changes**
(4 breaking releases in the last ~10), a 146 kLOC dependency of which pdum needs
maybe 12%, and it contributes **nothing** to the parts of pdum that are actually
novel (type-keyed caching, `typeof`, capture extraction, marshaling/ABI, WGSL
runtime). The honest split of the question is in §5; the recommendation is §6.

---

## 2. Maturity (state as of July 2026)

### 2.1 Version, cadence, churn

| Fact | Value | Source |
|---|---|---|
| Latest release | **v0.68.0, 2026-07-03** | PyPI / GitHub releases |
| Cadence | ~9 releases across ~3.5 months (v0.59→v0.68, spring–summer 2026) — roughly biweekly | GitHub releases + git tags |
| Versioning | 0.x; **no stability promise**; breaking changes flagged per-release | releases page |
| Recent breaking changes | v0.64 (error on direct mutation of op fields), v0.62 (LLVM 22 bump), v0.61 (BitEnumAttribute restructure), v0.59 (textual-format whitespace change) | releases page |
| Python support | >=3.10 through **3.14** (verified: installs and runs on 3.14) | pyproject.toml, my venv test |
| Runtime deps | only `immutabledict`, `ordered-set`, `typing-extensions` | pyproject.toml |
| Installed size | **14 MB** site-packages | measured |
| Import time | **65 ms** for `xdsl` + builtin/arith/func/scf dialects (Py 3.14, M-series Mac) | measured |
| License | Apache 2.0 with LLVM Exceptions | repo |
| Community | 562 stars, 168 forks, ~7 active maintainers, Zulip chat | GitHub |
| Pinned MLIR interop version | MLIR 22.1.2 | repo README |

Deprecation is done politely — my test hit a live example:
`PatternRewriter.replace_matched_op` now emits
`DeprecationWarning: Please use replace_op(op, new_op)`. There are 47
deprecation markers across 18 files in the current tree, i.e. the API is being
actively reshaped *with* migration paths, but reshaped nonetheless. Budget for
mechanical breakage on every upgrade, or pin and vendor.

### 2.2 Codebase size (measured on the clone)

| Component | LOC | Notes |
|---|---|---|
| **whole `xdsl/` package** | **146,289** | |
| `ir/` (Operation/Region/Block/SSAValue/Attribute) | 4,204 | `ir/core.py` alone is 2,723 |
| `irdl/` (declarative op/attr defs, constraints, assembly format) | 6,885 | `operations.py` 2,237; declarative format 2,493 |
| `parser/` + `printer.py` | 4,361 | full MLIR-compatible textual format |
| `pattern_rewriter.py` + `rewriter.py` + `builder.py` | 1,479 | |
| `traits.py` + `context.py` | 1,177 | |
| **"core you'd actually use"** | **~18,000** | the seven rows above |
| `dialects/` | 73,898 | **67+ dialects**: arith, scf, cf, func, builtin, gpu, memref, linalg, llvm, tensor, stencil, pdl, riscv, x86, wasm… |
| `transforms/` | 29,044 | 105 pass files (canonicalize, CSE, DCE, inlining, lowerings) |
| `interpreters/` | 5,155 | interpreter impls for the main dialects |
| `backend/` | 6,762 | riscv/x86 asm, CSL (Cerebras), **wgsl** (249 lines), mps |
| `frontend/` | 3,538 | includes `pyast`: a Python-AST → xDSL-IR frontend |

Notable: `xdsl/backend/wgsl/wgsl_printer.py` is a **249-line gpu-dialect→WGSL
printer** (singledispatch over op types). It is proof-of-concept grade — no
uniform-buffer story, `NotImplementedError` for most ops — but it demonstrates the
shape of a WGSL backend over the gpu dialect and is worth reading. There is also a
`frontend/pyast` package (Python `ast` → `func`/`scf`/`cf`/`symref` dialects) —
directly adjacent to pdum's frontend question, small enough to read in a sitting.

### 2.3 Community and users

Academic-led (Edinburgh, ETH Zurich, Cambridge, TU Berlin; Grosser/Steuwer
groups), ExCALIBUR-funded, CGO 2025 paper with artifact badges. Real users:
**Devito** and **PSyclone** (both HPC stencil DSLs) have compiler paths built on
xDSL with bespoke dialects — the PSyclone/xDSL path has run on 32,768 ARCHER2
cores and V100s (https://arxiv.org/pdf/2404.02218); a Snitch (RISC-V PULP)
compiler stack and a Cerebras CS-2 stencil flow are built on it
(https://xdsl.dev/events/2025-HiPEAC/). ASPLOS 2025 tutorial, HiPEAC 2025
workshop, MLIR winter school presence. This is a healthy, funded, multi-group
project — but its user base is HPC research compilers whose authors tolerate API
churn as a cost of doing business. There is no visible mass-market user whose
existence forces API stability.

**Safe to build on?** Safe in the "will exist and be maintained in 3 years"
sense: yes, very likely. Safe in the "my code compiles unchanged next quarter"
sense: no — pin the version.

---

## 3. Core model (verified against source)

### 3.1 IR structure

Classic MLIR: `Operation`s hold operand `SSAValue`s, produce `OpResult`s, and own
`Region`s; a `Region` holds `Block`s; a `Block` holds ops and `BlockArgument`s
(phi-replacement). Types are attributes (`TypeAttribute`); every `SSAValue` has
one. Key classes in `xdsl/ir/core.py`: `Dialect`, `Attribute` (frozen dataclass,
value-equal — important: **attributes hash/compare structurally**, so they can
participate in cache keys), `ParametrizedAttribute`, `Data[T]`, `SSAValue`,
`OpResult`, `BlockArgument`, `Operation`, `Block`, `Region`. Since v0.64,
direct mutation of op fields raises — mutation goes through `Rewriter` APIs,
which keeps use-def chains consistent.

### 3.2 IRDL: declarative op definitions

An op is a decorated class; fields are declared with typed helpers
(`xdsl/irdl/operations.py`):

```python
@irdl_op_definition
class AddiOp(SignlessIntegerBinaryOperation):   # real code, xdsl/dialects/arith.py
    name = "arith.addi"
    traits = traits_def(Pure(), Commutative(), ...)

class SignlessIntegerBinaryOperation(IRDLOperation, HasFolderInterface, abc.ABC):
    T: ClassVar = VarConstraint("T", signlessIntegerLike)   # type variable
    lhs = operand_def(T)
    rhs = operand_def(T)
    result = result_def(T)          # constraint solver enforces lhs:T == rhs:T == result:T
    assembly_format = "$lhs `,` $rhs attr-dict `:` type($result)"
```

Full helper set: `operand_def / var_operand_def / opt_operand_def`,
`result_def / var_* / opt_*`, `attr_def`, `prop_def`, `region_def`,
`successor_def`, `traits_def`. The decorator (`OpDef.from_pyrdl`) generates
accessors, `__init__`, the verifier, and — if `assembly_format` is given — the
parser and printer (`FormatProgram.from_str`). Custom semantic checks go in
`verify_()`. The constraint system (`irdl/constraints.py`, 1,346 lines) gives
`VarConstraint` type variables, unions (`AnyOf`), and parametrized-attribute
constraints — a small structural type checker you get for free per op.

**Boilerplate per op: ~4–8 lines.** Per dialect: one line —
`Dialect("pdum", [Op1, Op2, ...], [Attr1, ...])`.

### 3.3 Pattern rewriting

`xdsl/pattern_rewriter.py` (792 lines): subclass `RewritePattern`, implement
`match_and_rewrite(op, rewriter)`; the `@op_type_rewrite_pattern` decorator
dispatches on the parameter's type annotation. `PatternRewriter` provides
`replace_op`, `erase_op`, `insert_op`, `replace_all_uses_with`,
`replace_value_with_new_type`. `PatternRewriteWalker` runs a stack-based
`Worklist` to a fixpoint (newly created ops re-enqueued);
`GreedyRewritePatternApplier` composes pattern sets, with optional folding + DCE.
`TypeConversionPattern` (implement `convert_type(t)`) handles whole-IR type
rewrites — relevant to units auto-conversion later. There is also a
`HasCanonicalizationPatternsTrait` so ops carry their own canonicalizations, and
`HasFolderInterface` for constant folding (`fold()` on the op — see
`arith.AddiOp.py_operation`).

### 3.4 Interpreter

`xdsl/interpreter.py` (~1,100 lines): `Interpreter(module)` executes IR
directly. Implementations live in `InterpreterFunctions` subclasses; each op gets
an `@impl(OpType)` method receiving `(interpreter, op, args) -> results`;
registration via `@register_impls` + `interpreter.register_implementations(...)`.
Dispatch is `type(op)` keyed. `xdsl/interpreters/` ships impls for arith, scf,
func, memref, etc. **This is pdum's zero-dependency Python backend, free** —
verified in §3.6, including reading a *runtime* env value per call with no IR
change.

### 3.5 Textual format

Full MLIR-compatible parse/print round-tripping (`parser/` 3,655 + `printer.py`
706 lines), generic form always available, custom form via `assembly_format`.
This is xDSL's raison d'être (the CGO paper's "sidekick" thesis: interoperate
with MLIR through shared textual IR), and its practical value for pdum is
**golden-file testing and debuggability**: every stage of lowering is a
printable, parseable, diffable artifact — and an escape hatch to real MLIR/LLVM
if a native CPU backend is ever wanted.

### 3.6 The verified end-to-end exercise (the load-bearing evidence)

Run under xdsl 0.68.0 / Python 3.14; abridged only for width:

```python
@irdl_op_definition
class UniformOp(IRDLOperation):
    name = "pdum.uniform"
    slot = prop_def(IntegerAttr)        # ATTRIBUTE: compile-time constant (cache-key material)
    result = result_def(IntegerType)    # SSA VALUE: runtime data (re-marshaled per call)
    traits = traits_def(Pure())
    assembly_format = "$slot attr-dict `:` type($result)"

PdumDialect = Dialect("pdum", [UniformOp], [])
# ... build func @kernel: uniform(0) + (2+3); module.verify(); print → parse → verify (round-trip OK)
# ... FoldAddi pattern via PatternRewriteWalker: 2+3 folded to 5, uniform untouched
ENV = {0: 40}
@register_impls
class PdumImpls(InterpreterFunctions):
    @impl(UniformOp)
    def run_uniform(self, interp, op, args):
        return (ENV[op.slot.value.data],)
# interpret → (45,);  ENV[0] = 100 → re-run same compiled module → (105,)
```

Printed IR (exactly as emitted):

```mlir
builtin.module {
  func.func @kernel() -> i32 {
    %0 = pdum.uniform 0 : i64 : i32
    %1 = arith.constant 5 : i32          // after the rewrite walker
    %2 = arith.addi %0, %1 : i32
    func.return %2 : i32
  }
}
```

Total: ~70 lines for op definition + build + verify + round-trip + rewrite +
interpretation with a hot value re-run. That last line is the pdum thesis in
miniature: **the IR module is the compiled artifact; the env value changed; no
recompilation happened.**

---

## 4. Cost of adoption for pdum.dsl

### What you get free (vs the M0 reference asset, which had none of it)

| Free | Weight if hand-built |
|---|---|
| SSA IR with regions/blocks + use-def maintenance + post-order/dominance | 500–1,500 lines to do well |
| Per-op declarative verifier w/ type-variable constraints | 300–800 lines |
| Printer + parser, round-trip, golden-file testability | 1,000+ lines (usually skipped, then regretted) |
| Worklist rewrite driver + greedy applier + fold/DCE | 300–500 lines |
| Interpreter framework (= Python backend) | 300–600 lines |
| arith/scf/cf/func/builtin/gpu dialects + 105 transforms | thousands, amortized |
| MLIR escape hatch via textual format | priceless or worthless depending on ambition |

### What you still build 100% yourself (the actual pdum novelty)

- **The entire type-keyed caching layer**: `typeof`, structural fingerprints,
  `FnType`, env layout, generation/invalidation. xDSL has *no* opinion here.
- **Capture extraction** (`__closure__`/`co_freevars` walking) and the Python
  frontend policy (their `frontend/pyast` is a useful *reference*, not a drop-in:
  it has no closure-capture semantics and targets their dialects).
- **Marshaling/ABI**: logical value → N physical params, uniform-buffer packing,
  wgpu bind groups, CuPy kernel args. Nothing in xDSL.
- **A real WGSL backend** (theirs is a 249-line PoC without uniforms) and all
  other backends.
- **Autodiff/vmap** as dialect-to-dialect transforms (xDSL gives the rewrite
  machinery, not the math).

### Learning curve and friction

- Concept load: dialects, regions, blocks, ops, attributes vs values, traits,
  constraints, assembly format. For someone who knows MLIR: a weekend. From
  scratch: 1–2 weeks to fluency. Docs are decent (tutorial notebooks, a Toy
  compiler walkthrough, marimo notebooks) but API-reference-thin in places.
- Dependency weight is a non-issue at runtime (3 tiny pure-Python deps, 14 MB,
  65 ms import) but a real issue in *identity*: the "~1000-line kernel readable
  in a sitting" becomes "~1000 lines of ours on top of an 18 kLOC core we don't
  control, inside a 146 kLOC package."
- Churn tax: pin the version; expect mechanical fixes on each bump (they do ship
  deprecation shims first). The textual format itself changed whitespace in
  v0.59 — golden files are not immune.
- Performance: pure-Python IR objects. Fine for pdum's kernel sizes (tens–hundreds
  of ops) and irrelevant on the cached hot path, which never touches the IR at
  all. Do not build per-frame IR walks (M0's `flatten` mistake) on any IR,
  theirs or ours.

---

## 5. MLIR concepts worth stealing even without the dependency

Ranked by fit to pdum's desiderata:

1. **The value/attribute split.** Runtime data flows as SSA values (operands);
   compile-time constants are attributes on ops, and attributes are immutable,
   structurally-hashable values. This *is* pdum's uniform-vs-`Literal`
   distinction, made syntactic: a capture is an SSA block argument / `pdum.uniform`
   op (never in the cache key's value part); a `Literal`-lifted capture is an
   attribute (in the key by construction, because attributes are part of the IR's
   identity). Steal unconditionally.
2. **Dialects as the unit of extension.** A dialect = a named set of ops +
   attributes registered into a `Context`. The M0 fault line "core imports the
   WGSL intrinsic tables" dissolves: `pdum` core dialect, `pdum_wgsl` dialect,
   units dialect, einops dialect — each a plug-in. The frontend targets the core
   dialect only. This is the prime directive (incremental extensibility) as an
   architecture.
3. **Progressive lowering.** Frontend → high-level dialect → per-backend dialect →
   text emission, each step a small pass over the same infrastructure. Batteries
   economics (desiderata §7.4) falls out: define `pdum.mean` once at high level,
   lower it to loops/intrinsics per backend in a shared pass, override per backend
   only where needed.
4. **Regions for structured control flow.** `scf.if`/`scf.for` hold nested
   regions and *yield values* — no CFG needed for the WGSL/CUDA structured
   subset, and vastly easier autodiff than basic blocks. This is the single
   biggest fix for M0's "expression tree can't grow control flow" fault line.
   pdum likely never needs unstructured CFG (`cf`) at all.
5. **Traits/interfaces on ops.** `Pure`, `Commutative`, `HasFolder`,
   `IsTerminator`, memory-effect traits — passes query capabilities instead of
   switching on op names, so new ops work with old passes. Cheap to steal: a
   trait is a class attribute set on the op.
6. **Declarative op definition (IRDL-lite).** Even a homemade IR should define
   ops as data (operand count, types-as-constraints, attrs) and generate the
   verifier/constructor, not hand-write per-op classes with ad-hoc checks.
7. **A printable/parseable textual form.** Even print-only (no parser) buys
   golden-file tests and debugging. Print-only is ~150 lines.
8. **Worklist-to-fixpoint pattern application** with "new ops get re-enqueued"
   semantics — ~100 lines, and better than a hand-rolled recursive rewriter.

Skippable for pdum: successors/unstructured CFG, the constraint-solver
generality, declarative assembly-format compiler, symbol tables (maybe), 95% of
the dialects.

---

## 6. Honest assessment: embed xDSL, or purpose-built IR with stolen concepts?

### Strongest case FOR embedding xDSL

- The novel 1,000 lines of pdum are caching/marshaling/frontend — **none of which
  overlap xDSL**. Embedding means the 1,000 lines you write are *all* thesis, and
  the commodity compiler plumbing (verifier, printer, parser, rewriter,
  interpreter, arith/scf semantics) is professionally maintained by someone else.
  The §3.6 exercise shows the marginal cost of a pdum dialect is trivial.
- The interpreter is a free, always-correct Python backend and reference
  semantics for every op — the desiderata's "zero-dependency floor" nearly free.
- Golden-file round-trip testing from day one; the Devito/PSyclone precedent
  proves DSL-on-xDSL works at real scale; the MLIR escape hatch keeps a
  native-CPU future open without designing for it.
- The hot loop never touches the IR, so xDSL's Python-object overhead is confined
  to the once-per-type-signature compile path, where it is irrelevant.
- Runtime footprint is genuinely small (3 pure-Python deps, 65 ms import).

### Strongest case FOR a purpose-built ~300–600 line IR

- **Churn sovereignty.** xDSL is 0.x, breaks something roughly monthly, and its
  API-stability incentives come from HPC research users, not from products.
  pdum's cache-correctness invariants would sit on a foundation that reserves the
  right to change equality/hashing/printing semantics under it (v0.59 changed the
  textual format; v0.64 changed the mutation model).
- **Fit.** pdum needs ~15 ops, one region construct (`if`/`for` with yields),
  value/attribute split, a verifier, a printer, and a fixpoint rewriter. That is
  genuinely ~300–600 lines *when you already know the MLIR shape* — and this
  research means you do. The 18 kLOC core buys generality (variadics, constraint
  solving, unstructured CFG, MLIR-syntax fidelity) pdum doesn't need.
- **Identity and pedagogy.** "A tiny kernel readable in a sitting" is a stated
  aesthetic. A homemade IR keeps `typeof`→FnType→IR→backend traceable end-to-end
  in one repo with no foreign abstractions leaking in (IRDL constraint errors,
  `OpDef.from_pyrdl` metaclass magic in stack traces).
- **The cache key is the crown jewel.** Owning the IR means owning exactly what
  participates in structural identity (code object, env types, attributes) with
  no risk of a dependency's `__eq__`/`__hash__` semantics drifting.
- Migration insurance is cheap: if the homemade IR adopts the *concepts* (ops,
  regions, attributes, dialect-namespaced names) and an MLIR-flavored printed
  form, a later port onto xDSL/MLIR is mechanical, not architectural.

### Recommendation

**Steal the concept set wholesale; do not take the dependency for the kernel —
but keep xDSL as the explicitly-named compatibility target.** Concretely: build
the purpose-built IR with MLIR shape (ops/attrs/regions/SSA, dialect-namespaced
op names, value/attribute split, traits, a printed textual form that a human who
knows MLIR can read), sized ~300–600 lines including verifier and printer. Write
it so that op definitions are declarative data — which keeps open the tested
escape route of re-hosting the dialects on xDSL later if control flow, autodiff,
or a native backend outgrow the homemade core. Adopt xDSL immediately for one
*non-kernel* role where it is unbeatable and churn doesn't matter: as a dev-time
reference/testing harness (interpret pdum programs via a pdum→xDSL translation to
cross-check backend outputs), behind an optional extra, pinned.

The deciding factors: (a) pdum's identity *is* the tiny readable kernel plus
total ownership of cache-key semantics; (b) the delta xDSL provides over a
concept-faithful homemade IR is largest exactly in the areas pdum doesn't need
(MLIR fidelity, dialect breadth, unstructured CFG); (c) the 0.x churn tax is
recurring, while the cost of writing the small IR is paid once and is now — after
this research — well-understood. If the team's appetite changes (e.g. autodiff
proves hard and MLIR's `enzyme`/linalg ecosystems beckon), the MLIR-shaped IR and
textual form make xDSL adoption a refactor, not a rewrite.

---

## Design lessons for pdum.dsl

1. **Adopt the value/attribute split as the IR's constitutional principle.**
   Runtime captures (uniforms) are SSA values/operands; `Literal`-lifted captures
   are attributes. Attributes are immutable and structurally hashable, so "what's
   in the cache key" has a syntactic answer: the IR structure including
   attributes, never operand *values*. This unifies the M0 uniform-vs-constant
   distinction with the caching thesis in one mechanism.
2. **Make dialects the extension unit.** Core dialect (`pdum.*`) that the frontend
   targets; backend dialects (`pdum_wgsl.*`, `pdum_cuda.*`) and feature dialects
   (units, einops) registered into a context. This directly fixes M0's
   core→WGSL-intrinsics dependency inversion, and gives mini-languages (t-string
   sub-DSLs) a defined landing zone: each compiles to ops in some dialect.
3. **Use regions with value-yields for control flow, not a CFG.** `scf.if`/
   `scf.for`-style ops with nested single-block regions cover the entire
   WGSL/CUDA structured subset, keep autodiff tractable, and grow M0's expression
   tree without ever introducing basic blocks.
4. **Define ops declaratively (IRDL-lite, ~100 lines of infrastructure):** name,
   operand constraints, result constraints, attributes, traits — generate
   constructor + verifier from the declaration. Per-op cost must stay at the 4–8
   lines xDSL achieves, or incremental extensibility dies of boilerplate.
5. **Ship a textual printer (parser optional) from day one** and do golden-file
   tests on printed IR at every lowering stage. Keep the syntax MLIR-flavored so
   the xDSL/MLIR escape hatch stays a refactor, not a rewrite.
6. **Steal the worklist rewrite driver** (~100 lines: stack worklist, patterns
   dispatched by op type, new/modified ops re-enqueued, run to fixpoint) and
   attach canonicalization patterns and fold functions to ops via traits, so
   passes never switch on op names.
7. **Keep the IR entirely off the hot path.** In both xDSL and any homemade IR,
   per-call work must be: fingerprint types → hit cache → marshal values. IR
   exists only between cache miss and artifact; M0's per-frame `flatten` must not
   be reincarnated on a fancier IR.
8. **Use xDSL as a pinned, optional dev-time oracle:** a small pdum-IR→xDSL
   translator plus its interpreter gives a reference executor for
   differential-testing every backend — exploiting xDSL where it's strongest
   without coupling the kernel to its churn.
