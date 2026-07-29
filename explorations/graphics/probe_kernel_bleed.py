"""Probe: which tensor-layer vocabulary is reachable inside @compute bodies today?

Each probe builds a tiny kernel using one construct and reports lowered/refused.
"""


import numpy as np

from pdum.tl import Tensor, compute, global_idx

N = 8


def T(arr, names=("i",)):
    return Tensor.from_numpy(np.asarray(arr, dtype=np.float64), names)


def probe(name, make_kernel):
    x = T(np.arange(N, dtype=np.float64))
    y = T(np.zeros(N))
    try:
        k = make_kernel()
        k(x, y)
        print(f"LOWERS+RUNS  {name}: y={y.to_numpy()[:4]}...")
    except Exception as e:
        msg = str(e).replace("\n", " ")[:140]
        print(f"REFUSES      {name}: {type(e).__name__}: {msg}")


def k_flip():
    @compute
    def k(x, y):
        (i,) = global_idx("i")
        z = x.flip("i")
        y[i] = z[i]

    return k


def k_rename_method():
    @compute
    def k(x, y):
        (i,) = global_idx("i")
        z = x.rename(i="j")
        y[i] = x[i] + 0.0 * z.rename(j="i")[i]

    return k


def k_shift():
    @compute
    def k(x, y):
        (i,) = global_idx("i")
        z = x.shift(i=1)
        y[i] = z[i]

    return k


def k_select():
    @compute
    def k(x, y):
        (i,) = global_idx("i")
        z = x.select(i=0)  # selecting the only dim -> scalar-ish view
        y[i] = x[i] + 0.0 * z

    return k


def k_pad():
    @compute
    def k(x, y):
        (i,) = global_idx("i")
        z = x.pad(i=(1, 1), fill=0.0)
        y[i] = z[i]

    return k


def k_reduce_call():
    from pdum.tl import red

    @compute
    def k(x, y):
        (i,) = global_idx("i")
        s = red.sum(x, "i")
        y[i] = s

    return k


def k_if_stmt():
    @compute
    def k(x, y):
        (i,) = global_idx("i")
        v = x[i]
        if v > 3.0:
            y[i] = v
        else:
            y[i] = -v

    return k


def k_ifexp():
    @compute
    def k(x, y):
        (i,) = global_idx("i")
        v = x[i]
        y[i] = v if v > 3.0 else -v

    return k


def k_while():
    @compute
    def k(x, y):
        (i,) = global_idx("i")
        v = x[i]
        while v > 1.0:
            v = v - 1.0
        y[i] = v

    return k


probe("layout: flip in body", k_flip)
probe("layout: rename(kw) in body", k_rename_method)
probe("layout: shift in body", k_shift)
probe("layout: select in body", k_select)
probe("layout: pad in body", k_pad)
probe("reduce: red.sum in body", k_reduce_call)
probe("control: if statement", k_if_stmt)
probe("control: IfExp", k_ifexp)
probe("control: while", k_while)
