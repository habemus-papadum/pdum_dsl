"""The simplest compute kernel: C = A + B, elementwise.

Loads at the thread coordinates give the whole view; `+` is
pointwise-with-refusal; the store is the kernel's one effect."""

import numpy as np
from pdum.tl import Tensor, compute, thread_idx


@compute
def add_mat(A, B, C):
    y, x = thread_idx("y", "x")
    C[y, x] = A[y, x] + B[y, x]


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    A = Tensor.from_numpy(rng.standard_normal((3, 4)), ("y", "x"))
    B = Tensor.from_numpy(rng.standard_normal((3, 4)), ("y", "x"))
    C = Tensor.from_numpy(np.zeros((3, 4)), ("y", "x"))
    add_mat(A, B, C)
    np.testing.assert_allclose(C.to_numpy(), A.to_numpy() + B.to_numpy(), rtol=1e-12)
    print("add_mat OK:", C.to_numpy()[0])
