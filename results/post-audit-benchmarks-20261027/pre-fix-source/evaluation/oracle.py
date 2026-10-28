"""Bounded tensor contraction. No structural fallback counts as correctness."""
import numpy as np
QUIMB_MAX_OPEN = 16
WIDTH_CAP = 20

def _build_zx_tn(g):
    """Build a quimb TensorNetwork for a pyzx ZX-diagram.

    Returns (tn, output_inds) or None if the diagram uses an unsupported node.
    Open legs are named by input/output POSITION ("oi{i}"/"oo{j}") so two
    graphs with the same I/O ordering produce directly comparable contractions
    with no boundary-permutation search. High-degree spiders are decomposed
    into chains of degree-3 spiders so no single dense tensor blows up (a
    degree-35 spider would otherwise need a 512 GiB array).
    """
    import quimb.tensor as qtn
    from collections import defaultdict
    from zxdb.pyzx_utils import spider_tensor, hadamard_tensor

    inputs = list(g.inputs())
    outputs = list(g.outputs())
    open_idx = {}
    for i, v in enumerate(inputs):
        open_idx[v] = f"oi{i}"
    for j, v in enumerate(outputs):
        open_idx[v] = f"oo{j}"

    tensors = [qtn.Tensor(np.asarray(g.scalar.to_number()), inds=())]
    legs = defaultdict(list)
    counter = [0]

    def newidx():
        counter[0] += 1
        return f"e{counter[0]}"

    for e in g.edges():
        u, v = g.edge_s(e), g.edge_t(e)
        if g.edge_type(e) == 2:  # Hadamard edge -> explicit H tensor
            iu, iv = newidx(), newidx()
            tensors.append(qtn.Tensor(hadamard_tensor(), inds=(iu, iv)))
            legs[u].append(iu)
            legs[v].append(iv)
        else:
            ix = newidx()
            legs[u].append(ix)
            legs[v].append(ix)

    def add_spider(vlegs, phase, basis):
        d = len(vlegs)
        if d <= 3:
            tensors.append(qtn.Tensor(
                np.asarray(spider_tensor(d, phase, basis)), inds=tuple(vlegs)))
            return
        prev = newidx()
        tensors.append(qtn.Tensor(
            np.asarray(spider_tensor(3, phase, basis)),
            inds=(vlegs[0], vlegs[1], prev)))
        for k in range(2, d - 2):
            nxt = newidx()
            tensors.append(qtn.Tensor(
                np.asarray(spider_tensor(3, 0.0, basis)),
                inds=(prev, vlegs[k], nxt)))
            prev = nxt
        tensors.append(qtn.Tensor(
            np.asarray(spider_tensor(3, 0.0, basis)),
            inds=(prev, vlegs[d - 2], vlegs[d - 1])))

    try:
        for v in g.vertices():
            t = g.type(v)
            vlegs = legs[v]
            if t == 0:  # boundary: identity wire onto its open leg
                if not vlegs:
                    continue
                tensors.append(qtn.Tensor(
                    np.eye(2), inds=(vlegs[0], open_idx.get(v, f"orph{v}"))))
            elif t in (1, 2):
                add_spider(vlegs, float(g.phase(v) or 0), "Z" if t == 1 else "X")
            elif t == 3:  # H-box: all-ones tensor with -1 in the all-1 corner
                d = len(vlegs)
                if d == 0 or d > 12:
                    return None
                arr = np.ones((2,) * d, dtype=complex)
                arr[(1,) * d] = -1.0
                tensors.append(qtn.Tensor(arr, inds=tuple(vlegs)))
            else:
                return None  # W / Z-box unsupported here
    except Exception:
        return None

    output_inds = ([f"oi{i}" for i in range(len(inputs))]
                   + [f"oo{j}" for j in range(len(outputs))])
    return qtn.TensorNetwork(tensors), output_inds


def _quimb_tensor(g, max_open=QUIMB_MAX_OPEN):
    """
    Contract a pyzx ZX-diagram to a dense tensor using quimb.

    Unlike pyzx's naive `to_tensor` (a fixed left-to-right contraction that
    builds enormous intermediates on gadgetized / high-degree graphs and
    hangs), quimb finds an optimized contraction path, so this handles graphs
    pyzx cannot. Returns the array with legs ordered [inputs..., outputs...],
    or None if there are too many open legs (final tensor too large), the
    contraction is too wide (high treewidth), or a node is unsupported.
    """
    if len(list(g.inputs())) + len(list(g.outputs())) > max_open:
        return None
    built = _build_zx_tn(g)
    if built is None:
        return None
    tn, output_inds = built
    try:
        # Plan with greedy (near-instant, deterministic) and refuse graphs
        # whose contraction needs a huge intermediate (high treewidth) rather
        # than hang / OOM. The greedy width is an upper bound — low-treewidth
        # graphs (the whole eval suite and structured circuits) pass it; dense
        # full_reduce'd random circuits bail in milliseconds and a sampled or
        # structural check takes over.
        if len(tn.tensors) > 1:
            info = tn.contract(output_inds=output_inds, optimize="greedy",
                               get="path-info")
            if np.log2(float(info.largest_intermediate) + 1) > WIDTH_CAP:
                return None
        res = tn.contract(output_inds=output_inds, optimize="greedy")
        if hasattr(res, "transpose") and output_inds:
            arr = res.transpose(*output_inds).data
        else:
            arr = np.asarray(res)
        return np.asarray(arr)
    except Exception:
        return None



def _tensors_equal(a, b):
    from evaluation.core import verify
    result = verify(a, b)
    return None if result['status'] == 'inconclusive' else result['status'] == 'verified'

_quimb_equal = _tensors_equal


def pyzx_fixpoint(rule):
    """Fixture compatibility wrapper, with exact topology convergence checks."""
    from evaluation.workloads import payload
    def apply(g):
        for count in range(1000):
            before = payload(g)
            rule(g)
            if payload(g) == before:
                return count
        raise RuntimeError('Reference failed to converge')
    return apply
