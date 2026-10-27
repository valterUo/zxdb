"""Small scalable ZX rewrite patterns; n is pattern count or fan size."""
import random
from fractions import Fraction
import pyzx as zx

B, Z, X = zx.VertexType.BOUNDARY, zx.VertexType.Z, zx.VertexType.X
S, H = zx.EdgeType.SIMPLE, zx.EdgeType.HADAMARD


# --------------------------------------------------------------- generators

def g_spider_fusion(n):
    """n disjoint fusable Z-Z simple-edge pairs (batched by the DB query)."""
    g = zx.Graph(); ins = []; outs = []
    for _ in range(n):
        bi = g.add_vertex(B); z1 = g.add_vertex(Z, phase=Fraction(1, 4))
        z2 = g.add_vertex(Z, phase=Fraction(1, 4)); bo = g.add_vertex(B)
        g.add_edge((bi, z1), S); g.add_edge((z1, z2), S); g.add_edge((z2, bo), S)
        ins.append(bi); outs.append(bo)
    g.set_inputs(tuple(ins)); g.set_outputs(tuple(outs)); return g


def g_identity(n):
    """n disjoint degree-2 phase-0 identity spiders."""
    g = zx.Graph(); ins = []; outs = []
    for _ in range(n):
        bi = g.add_vertex(B); z = g.add_vertex(Z, phase=Fraction(0)); bo = g.add_vertex(B)
        g.add_edge((bi, z), S); g.add_edge((z, bo), S)
        ins.append(bi); outs.append(bo)
    g.set_inputs(tuple(ins)); g.set_outputs(tuple(outs)); return g


def g_hadamard(n):
    """n disjoint H - Z(0) - H chains (even Hadamard path -> cancels)."""
    g = zx.Graph(); ins = []; outs = []
    for _ in range(n):
        bi = g.add_vertex(B); z = g.add_vertex(Z, phase=Fraction(0)); bo = g.add_vertex(B)
        g.add_edge((bi, z), H); g.add_edge((z, bo), H)
        ins.append(bi); outs.append(bo)
    g.set_inputs(tuple(ins)); g.set_outputs(tuple(outs)); return g


def g_lcomp(n):
    """One interior Z(pi/2) with n Z(pi/4) leaves on Hadamard edges. One local
    complementation removes the centre and creates K_n (~n^2/2 H-edges)."""
    g = zx.Graph()
    c = g.add_vertex(Z, phase=Fraction(1, 2))
    for _ in range(n):
        leaf = g.add_vertex(Z, phase=Fraction(1, 4))
        g.add_edge((c, leaf), H)
    g.set_inputs(()); g.set_outputs(()); return g


def g_pivot(n):
    """Two interior Pauli Z-spiders on a Hadamard edge, each with n exclusive
    Z(pi/4) leaves (H-edges). One pivot toggles the n x n bipartite edges."""
    g = zx.Graph()
    a = g.add_vertex(Z, phase=Fraction(0)); b = g.add_vertex(Z, phase=Fraction(0))
    g.add_edge((a, b), H)
    for _ in range(n):
        la = g.add_vertex(Z, phase=Fraction(1, 4)); g.add_edge((a, la), H)
        lb = g.add_vertex(Z, phase=Fraction(1, 4)); g.add_edge((b, lb), H)
    g.set_inputs(()); g.set_outputs(()); return g


def g_pivot_gadget(n):
    """Interior Pauli Z(pi) - Hadamard - interior non-Pauli Z(pi/4), each with
    n Z(pi/4) leaves. One pivot-gadget application."""
    g = zx.Graph()
    a = g.add_vertex(Z, phase=Fraction(1)); b = g.add_vertex(Z, phase=Fraction(1, 4))
    g.add_edge((a, b), H)
    for _ in range(n):
        la = g.add_vertex(Z, phase=Fraction(1, 4)); g.add_edge((a, la), H)
        lb = g.add_vertex(Z, phase=Fraction(1, 4)); g.add_edge((b, lb), H)
    g.set_inputs(()); g.set_outputs(()); return g


def g_pivot_boundary(n):
    """Interior Pauli Z(pi) - Hadamard - Z spider that touches one boundary,
    each with n Z(pi/4) leaves. One pivot-boundary application."""
    g = zx.Graph(); ins = []
    a = g.add_vertex(Z, phase=Fraction(1))
    w = g.add_vertex(Z, phase=Fraction(1, 4))
    g.add_edge((a, w), H)
    bnd = g.add_vertex(B); g.add_edge((w, bnd), S); ins.append(bnd)
    for _ in range(n):
        la = g.add_vertex(Z, phase=Fraction(1, 4)); g.add_edge((a, la), H)
        lw = g.add_vertex(Z, phase=Fraction(1, 4)); g.add_edge((w, lw), H)
    g.set_inputs(tuple(ins)); g.set_outputs(()); return g


def g_gadget_fusion(n):
    """n phase gadgets sharing the SAME two targets -> all fuse into one."""
    g = zx.Graph()
    t1 = g.add_vertex(Z, phase=Fraction(0)); t2 = g.add_vertex(Z, phase=Fraction(0))
    for _ in range(n):
        axel = g.add_vertex(Z, phase=Fraction(0))
        leaf = g.add_vertex(Z, phase=Fraction(1, 4))
        g.add_edge((axel, leaf), H)
        g.add_edge((axel, t1), H); g.add_edge((axel, t2), H)
    g.set_inputs(()); g.set_outputs(()); return g


def g_supplementarity(n):
    """Two non-Clifford Z-spiders (pi/4, 3pi/4) sharing n common neighbours;
    one supplementarity removes both and adds pi to the n neighbours."""
    g = zx.Graph()
    v = g.add_vertex(Z, phase=Fraction(1, 4)); w = g.add_vertex(Z, phase=Fraction(3, 4))
    for _ in range(n):
        u = g.add_vertex(Z, phase=Fraction(1, 4))
        g.add_edge((v, u), H); g.add_edge((w, u), H)
    g.set_inputs(()); g.set_outputs(()); return g


def g_copy(n):
    """Z(pi/4) hub with one X(pi) copy-leaf (simple edge) and n other Z leaves;
    one copy propagates the leaf through the hub to all n neighbours."""
    g = zx.Graph()
    hub = g.add_vertex(Z, phase=Fraction(1, 4))
    leaf = g.add_vertex(X, phase=Fraction(1)); g.add_edge((hub, leaf), S)
    for _ in range(n):
        nb = g.add_vertex(Z, phase=Fraction(1, 4)); g.add_edge((hub, nb), H)
    g.set_inputs(()); g.set_outputs(()); return g


def g_bialgebra(n):
    """Phase-0 Z-centre - simple edge - phase-0 X-centre; the Z-centre has n
    phase-0 X-neighbours and the X-centre n phase-0 Z-neighbours. One bialgebra
    application creates the n x n bipartite edges."""
    g = zx.Graph()
    z0 = g.add_vertex(Z, phase=Fraction(0)); x0 = g.add_vertex(X, phase=Fraction(0))
    g.add_edge((z0, x0), S)
    for _ in range(n):
        xn = g.add_vertex(X, phase=Fraction(0)); g.add_edge((z0, xn), S)
        zn = g.add_vertex(Z, phase=Fraction(0)); g.add_edge((x0, zn), S)
    g.set_inputs(()); g.set_outputs(()); return g


def g_full_reduce(n):
    """Random CNOT+H+T circuit on n qubits, depth ~ 20*n."""
    random.seed(1000 + n)
    return zx.generate.CNOT_HAD_PHASE_circuit(
        qubits=n, depth=20 * n, p_had=0.2, p_t=0.25).to_graph()


