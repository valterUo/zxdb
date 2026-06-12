"""
Corner-case and average-case graphs for every rule in main_queries.json.

Each rule maps to:
  - db_method: name of the ZXdb method
  - pyzx: callable applying the pyzx counterpart
  - cases: list of (case_name, builder[, options]) where builder() returns a
    pyzx graph. options: require_iso=False relaxes structural equality for
    cases where rule application order may legitimately differ; tensor
    equivalence against the original then decides correctness.

Graphs are kept tiny (<= 3 qubits) so tensor comparison stays fast.
"""
import random
from fractions import Fraction
from functools import partial

import pyzx as zx
from pyzx.simplify import copy_simp as pyzx_copy_simp

from zxdb.generate import CNOT_HAD_PHASE_graph
from evaluation.harness import pyzx_fixpoint

B = zx.VertexType.BOUNDARY
Z = zx.VertexType.Z
X = zx.VertexType.X
S = zx.EdgeType.SIMPLE
H = zx.EdgeType.HADAMARD


def _g():
    return zx.Graph()


def _io(g, ins, outs):
    g.set_inputs(tuple(ins))
    g.set_outputs(tuple(outs))
    return g


# ---------------------------------------------------------------- spider fusion

def sf_chain3():
    """Three fusable Z spiders in a row -> single Z(1)."""
    g = _g()
    bi = g.add_vertex(B, 0, 0)
    z1 = g.add_vertex(Z, 0, 1, Fraction(1, 4))
    z2 = g.add_vertex(Z, 0, 2, Fraction(1, 4))
    z3 = g.add_vertex(Z, 0, 3, Fraction(1, 2))
    bo = g.add_vertex(B, 0, 4)
    g.add_edge((bi, z1), S)
    g.add_edge((z1, z2), S)
    g.add_edge((z2, z3), S)
    g.add_edge((z3, bo), S)
    return _io(g, [bi], [bo])


def sf_phase_wrap():
    """Phases sum past 2*pi: 3/2 + 3/4 = 9/4 -> must normalize to 1/4."""
    g = _g()
    bi = g.add_vertex(B, 0, 0)
    z1 = g.add_vertex(Z, 0, 1, Fraction(3, 2))
    z2 = g.add_vertex(Z, 0, 2, Fraction(3, 4))
    bo = g.add_vertex(B, 0, 3)
    g.add_edge((bi, z1), S)
    g.add_edge((z1, z2), S)
    g.add_edge((z2, bo), S)
    return _io(g, [bi], [bo])


def sf_xx():
    """X-X fusion."""
    g = _g()
    bi = g.add_vertex(B, 0, 0)
    x1 = g.add_vertex(X, 0, 1, Fraction(1, 2))
    x2 = g.add_vertex(X, 0, 2, Fraction(1, 2))
    bo = g.add_vertex(B, 0, 3)
    g.add_edge((bi, x1), S)
    g.add_edge((x1, x2), S)
    g.add_edge((x2, bo), S)
    return _io(g, [bi], [bo])


def sf_no_fuse_zx():
    """Z-X simple edge: must NOT fuse."""
    g = _g()
    bi = g.add_vertex(B, 0, 0)
    z1 = g.add_vertex(Z, 0, 1, Fraction(1, 4))
    x1 = g.add_vertex(X, 0, 2, Fraction(1, 4))
    bo = g.add_vertex(B, 0, 3)
    g.add_edge((bi, z1), S)
    g.add_edge((z1, x1), S)
    g.add_edge((x1, bo), S)
    return _io(g, [bi], [bo])


def sf_no_fuse_hadamard():
    """Z-Z Hadamard edge: must NOT fuse."""
    g = _g()
    bi = g.add_vertex(B, 0, 0)
    z1 = g.add_vertex(Z, 0, 1, Fraction(1, 4))
    z2 = g.add_vertex(Z, 0, 2, Fraction(1, 4))
    bo = g.add_vertex(B, 0, 3)
    g.add_edge((bi, z1), S)
    g.add_edge((z1, z2), H)
    g.add_edge((z2, bo), S)
    return _io(g, [bi], [bo])


def sf_hopf_via_fusion():
    """Fusing Z1,Z2 creates two parallel simple Z-X wires -> Hopf removes both."""
    g = _g()
    bi = g.add_vertex(B, 0, 0)
    z1 = g.add_vertex(Z, 0, 1)
    z2 = g.add_vertex(Z, 1, 1)
    x1 = g.add_vertex(X, 0, 2)
    bo = g.add_vertex(B, 0, 3)
    g.add_edge((bi, z1), S)
    g.add_edge((z1, z2), S)
    g.add_edge((z1, x1), S)
    g.add_edge((z2, x1), S)
    g.add_edge((x1, bo), S)
    return _io(g, [bi], [bo])


def sf_triangle():
    """Z triangle, all simple: collapses to a single spider; parallel edges
    become self-loops along the way."""
    g = _g()
    b1 = g.add_vertex(B, 0, 0)
    b2 = g.add_vertex(B, 1, 0)
    b3 = g.add_vertex(B, 2, 0)
    z1 = g.add_vertex(Z, 0, 1, Fraction(1, 4))
    z2 = g.add_vertex(Z, 1, 1, Fraction(1, 4))
    z3 = g.add_vertex(Z, 2, 1, Fraction(1, 2))
    g.add_edge((b1, z1), S)
    g.add_edge((b2, z2), S)
    g.add_edge((b3, z3), S)
    g.add_edge((z1, z2), S)
    g.add_edge((z2, z3), S)
    g.add_edge((z1, z3), S)
    return _io(g, [b1, b2], [b3])


def sf_keep_hadamard_neighbor():
    """Fusion must preserve the Hadamard edge to a third spider."""
    g = _g()
    bi = g.add_vertex(B, 0, 0)
    z1 = g.add_vertex(Z, 0, 1, Fraction(1, 4))
    z2 = g.add_vertex(Z, 0, 2, Fraction(1, 2))
    z3 = g.add_vertex(Z, 1, 2, Fraction(1, 4))
    bo = g.add_vertex(B, 1, 3)
    g.add_edge((bi, z1), S)
    g.add_edge((z1, z2), S)
    g.add_edge((z2, z3), H)
    g.add_edge((z3, bo), S)
    return _io(g, [bi], [bo])


# ---------------------------------------------------------------- identity removal

def id_simple_simple():
    """B - Z(0) - B -> direct wire."""
    g = _g()
    bi = g.add_vertex(B, 0, 0)
    z = g.add_vertex(Z, 0, 1, Fraction(0))
    bo = g.add_vertex(B, 0, 2)
    g.add_edge((bi, z), S)
    g.add_edge((z, bo), S)
    return _io(g, [bi], [bo])


def id_chain():
    """Two adjacent identities: B - Z(0) - Z(0) - B."""
    g = _g()
    bi = g.add_vertex(B, 0, 0)
    z1 = g.add_vertex(Z, 0, 1, Fraction(0))
    z2 = g.add_vertex(Z, 0, 2, Fraction(0))
    bo = g.add_vertex(B, 0, 3)
    g.add_edge((bi, z1), S)
    g.add_edge((z1, z2), S)
    g.add_edge((z2, bo), S)
    return _io(g, [bi], [bo])


def id_hadamard_combo():
    """simple + hadamard edges through the identity -> hadamard edge."""
    g = _g()
    bi = g.add_vertex(B, 0, 0)
    z1 = g.add_vertex(Z, 0, 1, Fraction(0))
    z2 = g.add_vertex(Z, 0, 2, Fraction(1, 4))
    bo = g.add_vertex(B, 0, 3)
    g.add_edge((bi, z1), S)
    g.add_edge((z1, z2), H)
    g.add_edge((z2, bo), S)
    return _io(g, [bi], [bo])


def id_double_hadamard():
    """hadamard + hadamard through the identity -> simple edge."""
    g = _g()
    bi = g.add_vertex(B, 0, 0)
    z1 = g.add_vertex(Z, 0, 1, Fraction(1, 4))
    z2 = g.add_vertex(Z, 0, 2, Fraction(0))
    z3 = g.add_vertex(Z, 0, 3, Fraction(1, 2))
    bo = g.add_vertex(B, 0, 4)
    g.add_edge((bi, z1), S)
    g.add_edge((z1, z2), H)
    g.add_edge((z2, z3), H)
    g.add_edge((z3, bo), S)
    return _io(g, [bi], [bo])


def id_x_identity():
    """Phase-0 degree-2 X spider is an identity too."""
    g = _g()
    bi = g.add_vertex(B, 0, 0)
    x = g.add_vertex(X, 0, 1, Fraction(0))
    bo = g.add_vertex(B, 0, 2)
    g.add_edge((bi, x), S)
    g.add_edge((x, bo), S)
    return _io(g, [bi], [bo])


def id_nonzero_phase():
    """Z(pi) degree-2: must NOT be removed."""
    g = _g()
    bi = g.add_vertex(B, 0, 0)
    z = g.add_vertex(Z, 0, 1, Fraction(1))
    bo = g.add_vertex(B, 0, 2)
    g.add_edge((bi, z), S)
    g.add_edge((z, bo), S)
    return _io(g, [bi], [bo])


def _id_triangle(e1, e2, e3):
    """Identity in a triangle: removal creates an edge parallel to an
    existing one. Edge types parameterized to hit all normalization paths."""
    g = _g()
    bi = g.add_vertex(B, 0, 0)
    z1 = g.add_vertex(Z, 0, 1, Fraction(1, 4))
    zid = g.add_vertex(Z, 1, 1, Fraction(0))
    z3 = g.add_vertex(Z, 0, 2, Fraction(1, 2))
    bo = g.add_vertex(B, 0, 3)
    g.add_edge((bi, z1), S)
    g.add_edge((z1, zid), e1)
    g.add_edge((zid, z3), e2)
    g.add_edge((z1, z3), e3)
    g.add_edge((z3, bo), S)
    return _io(g, [bi], [bo])


def id_in_triangle():
    """New simple edge parallel to existing Hadamard edge (s + h pair)."""
    return _id_triangle(S, S, H)


def id_in_triangle_hh():
    """New Hadamard edge parallel to existing Hadamard edge (pair cancels)."""
    return _id_triangle(H, S, H)


def id_in_triangle_ss():
    """New simple edge parallel to existing simple edge (collapse to one)."""
    return _id_triangle(S, S, S)


# ---------------------------------------------------------------- supplementarity

def supp_basic_type1():
    """Non-adjacent pair, same single neighbor, phases 1/4 + 3/4."""
    g = _g()
    bi = g.add_vertex(B, 0, 0)
    hub = g.add_vertex(Z, 0, 1, Fraction(0))
    t1 = g.add_vertex(Z, 0, 2, Fraction(1, 4))
    t2 = g.add_vertex(Z, 0, 3, Fraction(3, 4))
    bo = g.add_vertex(B, 0, 4)
    g.add_edge((bi, hub), S)
    g.add_edge((hub, bo), S)
    g.add_edge((hub, t1), H)
    g.add_edge((hub, t2), H)
    return _io(g, [bi], [bo])


def supp_type2_adjacent():
    """Adjacent pair with same external neighborhood."""
    g = _g()
    bi = g.add_vertex(B, 0, 0)
    hub = g.add_vertex(Z, 0, 1, Fraction(0))
    t1 = g.add_vertex(Z, 0, 2, Fraction(1, 4))
    t2 = g.add_vertex(Z, 0, 3, Fraction(3, 4))
    bo = g.add_vertex(B, 0, 4)
    g.add_edge((bi, hub), S)
    g.add_edge((hub, bo), S)
    g.add_edge((hub, t1), H)
    g.add_edge((hub, t2), H)
    g.add_edge((t1, t2), H)
    return _io(g, [bi], [bo])


def supp_type2_sum_zero():
    """Adjacent pair with phases summing to 0 mod 2 (1/4 + 7/4)."""
    g = _g()
    bi = g.add_vertex(B, 0, 0)
    hub = g.add_vertex(Z, 0, 1, Fraction(0))
    t1 = g.add_vertex(Z, 0, 2, Fraction(1, 4))
    t2 = g.add_vertex(Z, 0, 3, Fraction(7, 4))
    bo = g.add_vertex(B, 0, 4)
    g.add_edge((bi, hub), S)
    g.add_edge((hub, bo), S)
    g.add_edge((hub, t1), H)
    g.add_edge((hub, t2), H)
    g.add_edge((t1, t2), H)
    return _io(g, [bi], [bo])


def supp_no_fire_clifford():
    """Clifford phases (1/2 + 1/2): must NOT fire."""
    g = _g()
    bi = g.add_vertex(B, 0, 0)
    hub = g.add_vertex(Z, 0, 1, Fraction(0))
    t1 = g.add_vertex(Z, 0, 2, Fraction(1, 2))
    t2 = g.add_vertex(Z, 0, 3, Fraction(1, 2))
    bo = g.add_vertex(B, 0, 4)
    g.add_edge((bi, hub), S)
    g.add_edge((hub, bo), S)
    g.add_edge((hub, t1), H)
    g.add_edge((hub, t2), H)
    return _io(g, [bi], [bo])


def supp_no_fire_diff_neigh():
    """Different neighborhoods: must NOT fire."""
    g = _g()
    bi = g.add_vertex(B, 0, 0)
    hub1 = g.add_vertex(Z, 0, 1, Fraction(0))
    hub2 = g.add_vertex(Z, 1, 1, Fraction(0))
    t1 = g.add_vertex(Z, 0, 2, Fraction(1, 4))
    t2 = g.add_vertex(Z, 1, 2, Fraction(3, 4))
    bo = g.add_vertex(B, 0, 3)
    g.add_edge((bi, hub1), S)
    g.add_edge((hub1, hub2), H)
    g.add_edge((hub2, bo), S)
    g.add_edge((hub1, t1), H)
    g.add_edge((hub2, t2), H)
    return _io(g, [bi], [bo])


def supp_type1_diff_odd():
    """Non-adjacent pair with ODD DIFFERENCE (1/4 and 5/4): matches type 1
    but neighbors get NO phase update (pyzx only adds when the sum is odd)."""
    g = _g()
    bi = g.add_vertex(B, 0, 0)
    hub = g.add_vertex(Z, 0, 1, Fraction(1, 4))
    t1 = g.add_vertex(Z, 0, 2, Fraction(1, 4))
    t2 = g.add_vertex(Z, 0, 3, Fraction(5, 4))
    bo = g.add_vertex(B, 0, 4)
    g.add_edge((bi, hub), S)
    g.add_edge((hub, bo), S)
    g.add_edge((hub, t1), H)
    g.add_edge((hub, t2), H)
    return _io(g, [bi], [bo])


def supp_type2_diff_odd():
    """Adjacent pair with odd difference (1/4 and 5/4): matches type 2,
    no phase update (sum 3/2 is not even)."""
    g = _g()
    bi = g.add_vertex(B, 0, 0)
    hub = g.add_vertex(Z, 0, 1, Fraction(0))
    t1 = g.add_vertex(Z, 0, 2, Fraction(1, 4))
    t2 = g.add_vertex(Z, 0, 3, Fraction(5, 4))
    bo = g.add_vertex(B, 0, 4)
    g.add_edge((bi, hub), S)
    g.add_edge((hub, bo), S)
    g.add_edge((hub, t1), H)
    g.add_edge((hub, t2), H)
    g.add_edge((t1, t2), H)
    return _io(g, [bi], [bo])


def supp_two_neighbor_pair():
    """Pair sharing TWO common neighbors (phases must propagate to both)."""
    g = _g()
    bi = g.add_vertex(B, 0, 0)
    h1 = g.add_vertex(Z, 0, 1, Fraction(1, 4))
    h2 = g.add_vertex(Z, 1, 1, Fraction(1, 2))
    t1 = g.add_vertex(Z, 0, 2, Fraction(1, 4))
    t2 = g.add_vertex(Z, 0, 3, Fraction(3, 4))
    bo = g.add_vertex(B, 1, 0)
    g.add_edge((bi, h1), S)
    g.add_edge((h2, bo), S)
    g.add_edge((h1, h2), H)
    g.add_edge((h1, t1), H)
    g.add_edge((h1, t2), H)
    g.add_edge((h2, t1), H)
    g.add_edge((h2, t2), H)
    return _io(g, [bi], [bo])


# ---------------------------------------------------------------- copy rule

def copy_a0_mixed():
    """phase-0 leaf; hub has boundary and interior neighbors."""
    g = _g()
    bi = g.add_vertex(B, 0, 0)
    bo = g.add_vertex(B, 0, 4)
    w = g.add_vertex(Z, 0, 1, Fraction(1, 4))
    v = g.add_vertex(Z, 0, 2, Fraction(0))
    n1 = g.add_vertex(Z, 0, 3, Fraction(1, 4))
    n2 = g.add_vertex(Z, 1, 3, Fraction(3, 4))
    g.add_edge((bi, w), S)
    g.add_edge((w, bo), S)
    g.add_edge((w, v), H)
    g.add_edge((w, n1), H)
    g.add_edge((w, n2), H)
    return _io(g, [bi], [bo])


def copy_a1_interior():
    """phase-pi leaf: interior neighbors must get phase += 1."""
    g = _g()
    bi = g.add_vertex(B, 0, 0)
    bo = g.add_vertex(B, 0, 4)
    w = g.add_vertex(Z, 0, 1, Fraction(1, 4))
    v = g.add_vertex(Z, 0, 2, Fraction(1))
    n1 = g.add_vertex(Z, 0, 3, Fraction(1, 4))
    g.add_edge((bi, w), S)
    g.add_edge((w, bo), S)
    g.add_edge((w, v), H)
    g.add_edge((w, n1), H)
    return _io(g, [bi], [bo])


def copy_only_boundaries():
    """Hub has only boundary neighbors besides the leaf."""
    g = _g()
    bi = g.add_vertex(B, 0, 0)
    bo = g.add_vertex(B, 0, 3)
    w = g.add_vertex(Z, 0, 1, Fraction(1, 2))
    v = g.add_vertex(Z, 0, 2, Fraction(1))
    g.add_edge((bi, w), S)
    g.add_edge((w, bo), S)
    g.add_edge((w, v), H)
    return _io(g, [bi], [bo])


def copy_cascade():
    """First application turns an interior neighbor into a new copy leaf."""
    g = _g()
    bi = g.add_vertex(B, 0, 0)
    w2 = g.add_vertex(Z, 0, 1, Fraction(1, 4))
    n = g.add_vertex(Z, 0, 2, Fraction(0))
    w = g.add_vertex(Z, 0, 3, Fraction(1, 2))
    v = g.add_vertex(Z, 0, 4, Fraction(0))
    g.add_edge((bi, w2), S)
    g.add_edge((w2, n), H)
    g.add_edge((n, w), H)
    g.add_edge((w, v), H)
    return _io(g, [bi], [])


def copy_two_leaves():
    """Two copy leaves on the same hub: only one application is possible
    (the second leaf is consumed as an interior neighbor)."""
    g = _g()
    bi = g.add_vertex(B, 0, 0)
    w = g.add_vertex(Z, 0, 1, Fraction(1, 4))
    v1 = g.add_vertex(Z, 0, 2, Fraction(0))
    v2 = g.add_vertex(Z, 1, 2, Fraction(1))
    g.add_edge((bi, w), S)
    g.add_edge((w, v1), H)
    g.add_edge((w, v2), H)
    return _io(g, [bi], [])


def copy_simple_edge_leaf():
    """Z-leaf attached to a Z-spider by a SIMPLE edge: same colors across a
    simple wire, pyzx >= 0.10 does NOT fire (that is spider-fusion territory)."""
    g = _g()
    bi = g.add_vertex(B, 0, 0)
    bo = g.add_vertex(B, 0, 3)
    w = g.add_vertex(Z, 0, 1, Fraction(1, 4))
    v = g.add_vertex(Z, 0, 2, Fraction(1))
    g.add_edge((bi, w), S)
    g.add_edge((w, bo), S)
    g.add_edge((w, v), S)
    return _io(g, [bi], [bo])


def copy_x_leaf_simple():
    """X-leaf with phase pi attached to a Z-spider by a SIMPLE wire: classic
    state copy, fires with copies keeping the X color."""
    g = _g()
    bi = g.add_vertex(B, 0, 0)
    bo = g.add_vertex(B, 0, 3)
    w = g.add_vertex(Z, 0, 1, Fraction(1, 4))
    v = g.add_vertex(X, 0, 2, Fraction(1))
    g.add_edge((bi, w), S)
    g.add_edge((w, bo), S)
    g.add_edge((w, v), S)
    return _io(g, [bi], [bo])


def copy_no_fire_h_diff_colors():
    """X-leaf H-connected to a Z-spider: colors differ across a Hadamard
    wire, must NOT fire."""
    g = _g()
    bi = g.add_vertex(B, 0, 0)
    bo = g.add_vertex(B, 0, 3)
    w = g.add_vertex(Z, 0, 1, Fraction(1, 4))
    v = g.add_vertex(X, 0, 2, Fraction(1))
    g.add_edge((bi, w), S)
    g.add_edge((w, bo), S)
    g.add_edge((w, v), H)
    return _io(g, [bi], [bo])


# ---------------------------------------------------------------- local complementation

def _graphlike_base(n_spiders=2):
    """2-qubit graph-like skeleton: boundaries -> Z spiders via simple edges."""
    g = _g()
    bi1 = g.add_vertex(B, 0, 0)
    bi2 = g.add_vertex(B, 1, 0)
    z1 = g.add_vertex(Z, 0, 1, Fraction(0))
    z2 = g.add_vertex(Z, 1, 1, Fraction(0))
    bo1 = g.add_vertex(B, 0, 9)
    bo2 = g.add_vertex(B, 1, 9)
    g.add_edge((bi1, z1), S)
    g.add_edge((bi2, z2), S)
    g.add_edge((z1, bo1), S)
    g.add_edge((z2, bo2), S)
    _io(g, [bi1, bi2], [bo1, bo2])
    return g, z1, z2


def lc_basic():
    """Normal case: center(1/2) H-connected to two unconnected spiders ->
    spiders become connected, phases -= 1/2."""
    g, z1, z2 = _graphlike_base()
    c = g.add_vertex(Z, 2, 1, Fraction(1, 2))
    g.add_edge((c, z1), H)
    g.add_edge((c, z2), H)
    return g


def lc_toggle_off():
    """Neighbors already H-connected: lcomp removes that edge."""
    g, z1, z2 = _graphlike_base()
    c = g.add_vertex(Z, 2, 1, Fraction(1, 2))
    g.add_edge((c, z1), H)
    g.add_edge((c, z2), H)
    g.add_edge((z1, z2), H)
    return g


def lc_neg_phase():
    """Center with phase 3/2 (= -1/2): the old query only matched 0.5/-0.5
    and could never fire on this."""
    g, z1, z2 = _graphlike_base()
    c = g.add_vertex(Z, 2, 1, Fraction(3, 2))
    g.add_edge((c, z1), H)
    g.add_edge((c, z2), H)
    return g


def lc_three_neighbors():
    """Center with three neighbors: full triangle toggling."""
    g, z1, z2 = _graphlike_base()
    z3 = g.add_vertex(Z, 2, 2, Fraction(1, 4))
    g.add_edge((z1, z3), H)
    c = g.add_vertex(Z, 2, 1, Fraction(1, 2))
    g.add_edge((c, z1), H)
    g.add_edge((c, z2), H)
    g.add_edge((c, z3), H)
    return g


def lc_leaf_center():
    """Degree-1 center (phase-gadget tip with phase 1/2): pyzx applies lcomp;
    the old query dropped the row and never deleted the center."""
    g, z1, z2 = _graphlike_base()
    c = g.add_vertex(Z, 2, 1, Fraction(1, 2))
    g.add_edge((c, z1), H)
    return g


def lc_no_match():
    """No lcomp pattern at all: the old driver loop hung forever here."""
    g, z1, z2 = _graphlike_base()
    g.add_edge((z1, z2), H)
    g.set_phase(z1, Fraction(1, 4))
    return g


def lc_no_fire_boundary_center():
    """Center(1/2) with a simple edge to a boundary: NOT interior, must not
    fire (the old query fired and silently severed the boundary wire)."""
    g = _g()
    bi = g.add_vertex(B, 0, 0)
    c = g.add_vertex(Z, 0, 1, Fraction(1, 2))
    z1 = g.add_vertex(Z, 0, 2, Fraction(1, 4))
    z2 = g.add_vertex(Z, 1, 2, Fraction(3, 4))
    bo1 = g.add_vertex(B, 0, 3)
    bo2 = g.add_vertex(B, 1, 3)
    g.add_edge((bi, c), S)
    g.add_edge((c, z1), H)
    g.add_edge((c, z2), H)
    g.add_edge((z1, bo1), S)
    g.add_edge((z2, bo2), S)
    return _io(g, [bi], [bo1, bo2])


def lc_no_fire_simple_edge():
    """Center(1/2) with one SIMPLE edge to a Z spider: not graph-like at the
    center, must not fire."""
    g, z1, z2 = _graphlike_base()
    c = g.add_vertex(Z, 2, 1, Fraction(1, 2))
    g.add_edge((c, z1), S)
    g.add_edge((c, z2), H)
    return g


def lc_two_candidates_shared():
    """Two lcomp candidates sharing a neighbor: application order may differ
    between DB and pyzx, so only tensor equivalence is required."""
    g, z1, z2 = _graphlike_base()
    c1 = g.add_vertex(Z, 2, 1, Fraction(1, 2))
    c2 = g.add_vertex(Z, 3, 1, Fraction(3, 2))
    g.add_edge((c1, z1), H)
    g.add_edge((c1, z2), H)
    g.add_edge((c2, z2), H)
    g.add_edge((c2, z1), H)
    return g


# ---------------------------------------------------------------- pivot

def _pivot_base(phase_a, phase_b):
    """2-qubit graph-like skeleton plus an interior Pauli pair a-b where a is
    H-connected to z1 (qubit 0) and b to z2 (qubit 1)."""
    g, z1, z2 = _graphlike_base()
    a = g.add_vertex(Z, 2, 1, phase_a)
    b = g.add_vertex(Z, 3, 1, phase_b)
    g.add_edge((a, b), H)
    g.add_edge((a, z1), H)
    g.add_edge((b, z2), H)
    return g, a, b, z1, z2


def pv_basic_00():
    """Normal case: interior Pauli pair, phases 0/0."""
    g, *_ = _pivot_base(Fraction(0), Fraction(0))
    return g


def pv_mixed_01():
    """Phases 0/1: exposes the swapped phase propagation in the old query
    (a's exclusive neighbors must receive b's phase, not a's)."""
    g, *_ = _pivot_base(Fraction(0), Fraction(1))
    return g


def pv_both_pi():
    """Phases 1/1."""
    g, *_ = _pivot_base(Fraction(1), Fraction(1))
    return g


def pv_shared_neighbor():
    """A shared neighbor: gets a + b + pi and toggles against both groups."""
    g, a, b, z1, z2 = _pivot_base(Fraction(0), Fraction(1))
    c = g.add_vertex(Z, 4, 1, Fraction(1, 4))
    g.add_edge((a, c), H)
    g.add_edge((b, c), H)
    return g


def pv_no_fire_nonpauli():
    """a has phase 1/4: must NOT fire."""
    g, *_ = _pivot_base(Fraction(1, 4), Fraction(0))
    return g


def pv_no_fire_simple_edge():
    """Pauli pair connected by a SIMPLE edge: must NOT fire."""
    g, z1, z2 = _graphlike_base()
    a = g.add_vertex(Z, 2, 1, Fraction(0))
    b = g.add_vertex(Z, 3, 1, Fraction(1))
    g.add_edge((a, b), S)
    g.add_edge((a, z1), H)
    g.add_edge((b, z2), H)
    return g


def pv_isolated_pair():
    """Pauli pair with NO other neighbors: both deleted, nothing else
    (exercises the empty neighbor-group path through the toggle subqueries)."""
    g, z1, z2 = _graphlike_base()
    a = g.add_vertex(Z, 2, 1, Fraction(0))
    b = g.add_vertex(Z, 3, 1, Fraction(1))
    g.add_edge((a, b), H)
    return g


def pv_boundary_deg2():
    """Boundary pivot, minimal: b is degree-2 (one boundary + a)."""
    g = _g()
    bi = g.add_vertex(B, 0, 0)
    b = g.add_vertex(Z, 0, 1, Fraction(1))
    a = g.add_vertex(Z, 0, 2, Fraction(0))
    z1 = g.add_vertex(Z, 1, 2, Fraction(1, 4))
    bo = g.add_vertex(B, 1, 3)
    g.add_edge((bi, b), S)
    g.add_edge((b, a), H)
    g.add_edge((a, z1), H)
    g.add_edge((z1, bo), S)
    return _io(g, [bi], [bo])


def pv_boundary_general():
    """Boundary pivot where b has an extra interior neighbor: pyzx fires, the
    old DB single-spider query (degree-2 only) missed this entirely."""
    g = _g()
    bi = g.add_vertex(B, 0, 0)
    b = g.add_vertex(Z, 0, 1, Fraction(1))
    a = g.add_vertex(Z, 0, 2, Fraction(0))
    z1 = g.add_vertex(Z, 1, 2, Fraction(1, 4))
    z2 = g.add_vertex(Z, 1, 1, Fraction(3, 4))
    bo1 = g.add_vertex(B, 1, 3)
    bo2 = g.add_vertex(B, 2, 3)
    g.add_edge((bi, b), S)
    g.add_edge((b, a), H)
    g.add_edge((b, z2), H)
    g.add_edge((a, z1), H)
    g.add_edge((z1, bo1), S)
    g.add_edge((z2, bo2), S)
    return _io(g, [bi], [bo1, bo2])


def pv_no_fire_two_boundaries():
    """Both a and b touch boundaries: pyzx allows at most one boundary in a
    pivot, must NOT fire."""
    g = _g()
    bi = g.add_vertex(B, 0, 0)
    bo = g.add_vertex(B, 0, 3)
    a = g.add_vertex(Z, 0, 1, Fraction(0))
    b = g.add_vertex(Z, 0, 2, Fraction(1))
    g.add_edge((bi, a), S)
    g.add_edge((a, b), H)
    g.add_edge((b, bo), S)
    return _io(g, [bi], [bo])


# ---------------------------------------------------------------- pivot gadget

def _pivot_gadget_base(phase_j, phase_alpha):
    """Interior Pauli z_j H-connected to interior non-Pauli z_alpha; z_j sees
    z1, z_alpha sees z2 (both boundary-attached spiders)."""
    g, z1, z2 = _graphlike_base()
    zj = g.add_vertex(Z, 2, 1, phase_j)
    za = g.add_vertex(Z, 3, 1, phase_alpha)
    g.add_edge((zj, za), H)
    g.add_edge((zj, z1), H)
    g.add_edge((za, z2), H)
    return g, zj, za, z1, z2


def pg_basic():
    """Normal case: j = pi, alpha = 1/4."""
    g, *_ = _pivot_gadget_base(Fraction(1), Fraction(1, 4))
    return g


def pg_j_zero():
    """j = 0: the old query hardcoded the axis phase to pi and negated the tip,
    which is semantically equal but not isomorphic to pyzx output."""
    g, *_ = _pivot_gadget_base(Fraction(0), Fraction(1, 4))
    return g


def pg_shared_neighbor():
    """Shared neighbor between z_j and z_alpha."""
    g, zj, za, z1, z2 = _pivot_gadget_base(Fraction(1), Fraction(3, 4))
    c = g.add_vertex(Z, 4, 1, Fraction(1, 4))
    g.add_edge((zj, c), H)
    g.add_edge((za, c), H)
    return g


def pg_connected_neighbors():
    """z_j's and z_alpha's exclusive neighbors already H-connected: the
    bipartite edge must be toggled OFF (old query created a parallel edge)."""
    g, zj, za, z1, z2 = _pivot_gadget_base(Fraction(1), Fraction(1, 4))
    g.add_edge((z1, z2), H)
    return g


def pg_no_fire_both_pauli():
    """Both phases Pauli: that is a plain pivot, the gadget rule must NOT
    fire (the old query had no non-Pauli requirement on z_alpha)."""
    g, *_ = _pivot_gadget_base(Fraction(1), Fraction(1))
    return g


def pg_no_fire_tip():
    """z_alpha is a degree-1 phase-gadget tip: must NOT fire (the old query
    would re-gadgetize forever)."""
    g, z1, z2 = _graphlike_base()
    zj = g.add_vertex(Z, 2, 1, Fraction(1))
    tip = g.add_vertex(Z, 3, 1, Fraction(1, 4))
    g.add_edge((zj, z1), H)
    g.add_edge((zj, z2), H)
    g.add_edge((zj, tip), H)
    return g


def pg_no_fire_boundary():
    """z_j touches a boundary: not interior, must NOT fire."""
    g = _g()
    bi = g.add_vertex(B, 0, 0)
    zj = g.add_vertex(Z, 0, 1, Fraction(1))
    za = g.add_vertex(Z, 0, 2, Fraction(1, 4))
    z1 = g.add_vertex(Z, 1, 2, Fraction(1, 2))
    bo = g.add_vertex(B, 1, 3)
    g.add_edge((bi, zj), S)
    g.add_edge((zj, za), H)
    g.add_edge((za, z1), H)
    g.add_edge((z1, bo), S)
    return _io(g, [bi], [bo])


# ---------------------------------------------------------------- pivot boundary

def _pivot_boundary_base(phase_j, phase_alpha):
    """Interior Pauli z_j H-connected to z_alpha that holds one boundary.
    z_j also sees z1 (boundary-attached on qubit 0)."""
    g = _g()
    bi1 = g.add_vertex(B, 0, 0)
    bi2 = g.add_vertex(B, 1, 0)
    z1 = g.add_vertex(Z, 0, 1, Fraction(0))
    za = g.add_vertex(Z, 1, 1, phase_alpha)
    zj = g.add_vertex(Z, 2, 1, phase_j)
    bo1 = g.add_vertex(B, 0, 9)
    g.add_edge((bi1, z1), S)
    g.add_edge((z1, bo1), S)
    g.add_edge((bi2, za), S)
    g.add_edge((zj, za), H)
    g.add_edge((zj, z1), H)
    _io(g, [bi1, bi2], [bo1])
    return g, zj, za, z1


def pb_basic():
    """Normal case: j = pi, alpha = 1/2 (Clifford boundary spider)."""
    g, *_ = _pivot_boundary_base(Fraction(1), Fraction(1, 2))
    return g


def pb_j_zero():
    """j = 0: axis/tip phase convention check against pyzx."""
    g, *_ = _pivot_boundary_base(Fraction(0), Fraction(1, 2))
    return g


def pb_w_nonclifford():
    """z_alpha has a non-Clifford phase (1/4): pyzx still fires (fallback w)."""
    g, *_ = _pivot_boundary_base(Fraction(1), Fraction(1, 4))
    return g


def pb_shared_neighbor():
    """z_j and z_alpha share a neighbor."""
    g, zj, za, z1 = _pivot_boundary_base(Fraction(1), Fraction(1, 2))
    c = g.add_vertex(Z, 3, 1, Fraction(1, 4))
    g.add_edge((zj, c), H)
    g.add_edge((za, c), H)
    return g


def pb_no_fire_v_boundary():
    """z_j itself touches a boundary: must NOT fire."""
    g = _g()
    bi1 = g.add_vertex(B, 0, 0)
    bi2 = g.add_vertex(B, 1, 0)
    zj = g.add_vertex(Z, 0, 1, Fraction(1))
    za = g.add_vertex(Z, 1, 1, Fraction(1, 2))
    g.add_edge((bi1, zj), S)
    g.add_edge((bi2, za), S)
    g.add_edge((zj, za), H)
    return _io(g, [bi1, bi2], [])


# ---------------------------------------------------------------- gadget fusion

def _add_gadget(g, targets, leaf_phase, axel_phase=Fraction(0)):
    axel = g.add_vertex(Z, 5, 5, axel_phase)
    leaf = g.add_vertex(Z, 6, 6, leaf_phase)
    g.add_edge((axel, leaf), H)
    for t in targets:
        g.add_edge((axel, t), H)
    return axel, leaf


def gf_two_same_targets():
    """Normal case: two gadgets on the same targets -> phases add."""
    g, z1, z2 = _graphlike_base()
    _add_gadget(g, [z1, z2], Fraction(1, 4))
    _add_gadget(g, [z1, z2], Fraction(1, 4))
    return g


def gf_axel_pi():
    """One axel carries pi: that gadget's phase contributes NEGATED
    (old query summed raw phases)."""
    g, z1, z2 = _graphlike_base()
    _add_gadget(g, [z1, z2], Fraction(1, 4))
    _add_gadget(g, [z1, z2], Fraction(1, 4), Fraction(1))
    return g


def gf_single_axel_pi():
    """Single gadget with axel pi: pyzx normalizes (leaf negated, axel zeroed)
    even without a fusion partner."""
    g, z1, z2 = _graphlike_base()
    _add_gadget(g, [z1, z2], Fraction(1, 4), Fraction(1))
    return g


def gf_three_gadgets():
    """Three gadgets on the same targets collapse into one."""
    g, z1, z2 = _graphlike_base()
    _add_gadget(g, [z1, z2], Fraction(1, 4))
    _add_gadget(g, [z1, z2], Fraction(3, 4))
    _add_gadget(g, [z1, z2], Fraction(7, 4))
    return g


def gf_no_fuse_diff_targets():
    """Different target sets: must NOT fuse."""
    g, z1, z2 = _graphlike_base()
    _add_gadget(g, [z1], Fraction(1, 4))
    _add_gadget(g, [z1, z2], Fraction(1, 4))
    return g


def gf_no_fuse_clifford_leaf():
    """Leaf phases 1/2 are Clifford: pyzx does NOT treat these as gadgets
    (the old query had no phase restriction and fused them)."""
    g, z1, z2 = _graphlike_base()
    _add_gadget(g, [z1, z2], Fraction(1, 2))
    _add_gadget(g, [z1, z2], Fraction(1, 2))
    return g


def gf_no_fuse_pauli_leaf():
    """Leaf phase pi is copy-rule territory, not a gadget: must NOT fuse."""
    g, z1, z2 = _graphlike_base()
    _add_gadget(g, [z1, z2], Fraction(1))
    _add_gadget(g, [z1, z2], Fraction(1))
    return g


# ---------------------------------------------------------------- bialgebra

def _bialg_base(n_left, n_right, v0_phase=Fraction(0), v1_phase=Fraction(0),
                left_phase=Fraction(0), right_phase=Fraction(0)):
    """v0 Z-spider <- n_left X-neighbors (each boundary-attached);
    v1 X-spider <- n_right Z-neighbors (each boundary-attached);
    v0 - v1 simple edge."""
    g = _g()
    v0 = g.add_vertex(Z, 0, 4, v0_phase)
    v1 = g.add_vertex(X, 0, 5, v1_phase)
    g.add_edge((v0, v1), S)
    ins, outs = [], []
    lefts, rights = [], []
    for k in range(n_left):
        bb = g.add_vertex(B, k, 0)
        x = g.add_vertex(X, k, 1, left_phase)
        g.add_edge((bb, x), S)
        g.add_edge((x, v0), S)
        ins.append(bb)
        lefts.append(x)
    for k in range(n_right):
        bb = g.add_vertex(B, k, 9)
        z = g.add_vertex(Z, k, 8, right_phase)
        g.add_edge((bb, z), S)
        g.add_edge((z, v1), S)
        outs.append(bb)
        rights.append(z)
    _io(g, ins, outs)
    return g, v0, v1, lefts, rights


def ba_basic_k11():
    """Minimal: one X neighbor, one Z neighbor -> single new edge."""
    g, *_ = _bialg_base(1, 1)
    return g


def ba_k22():
    """Two neighbors each: 4 new bipartite edges."""
    g, *_ = _bialg_base(2, 2)
    return g


def ba_existing_edge():
    """A neighbor pair is already directly connected: pyzx >= 0.10 leaves
    that wire untouched (new spiders are inserted on the center wires)."""
    g, v0, v1, lefts, rights = _bialg_base(2, 1)
    g.add_edge((lefts[0], rights[0]), S)
    return g


def ba_pi_center():
    """v0 carries pi: pyzx >= 0.10 fires on PAULI centers and copies the
    pi onto the new spiders of the opposite side."""
    g, *_ = _bialg_base(1, 1, v0_phase=Fraction(1))
    return g


def ba_no_fire_nonpauli():
    """v0 carries 1/4: not Pauli, must NOT fire."""
    g, *_ = _bialg_base(1, 1, v0_phase=Fraction(1, 4))
    return g


def ba_no_fire_neighbor_phase():
    """A neighbor carries a phase: pyzx requires phase-0 neighbors."""
    g, *_ = _bialg_base(1, 1, left_phase=Fraction(1, 4))
    return g


def ba_no_fire_wrong_color():
    """v0 has a Z-neighbor (same color): must NOT fire."""
    g, v0, v1, lefts, rights = _bialg_base(1, 1)
    extra = g.add_vertex(Z, 5, 4, Fraction(0))
    g.add_edge((extra, v0), S)
    return g


def ba_no_fire_boundary_neighbor():
    """v0 is directly attached to a boundary: pyzx neighbors must be spiders."""
    g, v0, v1, lefts, rights = _bialg_base(1, 1)
    bb = g.add_vertex(B, 5, 0)
    g.add_edge((bb, v0), S)
    ins = list(g.inputs()) + [bb]
    g.set_inputs(tuple(ins))
    return g


# ---------------------------------------------------------------- hadamard cancellation

def _h_chain(n_edges, mid_phase=Fraction(0)):
    """z(1/4) -[H]- id -[H]- ... -[H]- z(1/2) with n_edges Hadamard edges and
    n_edges-1 phase-0 intermediates (mid_phase overrides the first one)."""
    g = _g()
    bi = g.add_vertex(B, 0, 0)
    left = g.add_vertex(Z, 0, 1, Fraction(1, 4))
    prev = left
    for k in range(n_edges - 1):
        mid = g.add_vertex(Z, 0, 2 + k, mid_phase if k == 0 else Fraction(0))
        g.add_edge((prev, mid), H)
        prev = mid
    right = g.add_vertex(Z, 0, 2 + n_edges, Fraction(1, 2))
    g.add_edge((prev, right), H)
    bo = g.add_vertex(B, 0, 3 + n_edges)
    g.add_edge((bi, left), S)
    g.add_edge((right, bo), S)
    return _io(g, [bi], [bo])


def hc_single_pair():
    """H - Z(0) - H collapses to a simple wire (= identity removal)."""
    return _h_chain(2)


def hc_chain4():
    """Four H edges through three identities: both sides give a simple wire."""
    return _h_chain(4)


def hc_chain3():
    """Odd chain: hadamard_cancel collapses only the even sub-path and leaves
    one identity; pyzx id_simp removes everything. Only semantics required."""
    return _h_chain(3)


def hc_no_fire_phase():
    """Intermediate carries pi: not an identity, nothing may fire."""
    return _h_chain(2, mid_phase=Fraction(1))


def hc_boundary_ends():
    """Path ends directly on boundaries: B -H- Z(0) -H- B."""
    g = _g()
    bi = g.add_vertex(B, 0, 0)
    mid = g.add_vertex(Z, 0, 1, Fraction(0))
    bo = g.add_vertex(B, 0, 2)
    g.add_edge((bi, mid), H)
    g.add_edge((mid, bo), H)
    return _io(g, [bi], [bo])


# ---------------------------------------------------------------- random (average) cases

def random_circuit(seed, qubits=3, depth=8, clifford=False):
    """Random CNOT+H+phase circuit as a raw ZX graph (deterministic seed)."""
    random.seed(seed)
    return CNOT_HAD_PHASE_graph(qubits, depth, p_had=0.25, p_t=0.3,
                                clifford=clifford)


def random_graphlike(seed, qubits=3, depth=8, clifford=False):
    """Random circuit brought to graph-like form (all Z, interior Hadamard
    edges) — the state interior rules operate on inside full_reduce."""
    g = random_circuit(seed, qubits, depth, clifford)
    pyzx_fixpoint(zx.spider_simp)(g)
    zx.to_gh(g)
    pyzx_fixpoint(zx.spider_simp)(g)
    pyzx_fixpoint(zx.id_simp)(g)
    return g


def _random_cases(builder, seeds, **kwargs):
    """Average-case entries; application order may differ between DB and
    pyzx, so only tensor equivalence is required."""
    return [(f"random_seed{s}", partial(builder, s, **kwargs),
             {"require_iso": False}) for s in seeds]


# ---------------------------------------------------------------- registry

RULES = {
    "spider_fusion": {
        "db_method": "spider_fusion",
        "pyzx": pyzx_fixpoint(zx.spider_simp),
        "cases": [
            ("chain3", sf_chain3),
            ("phase_wrap", sf_phase_wrap),
            ("xx", sf_xx),
            ("no_fuse_zx", sf_no_fuse_zx),
            ("no_fuse_hadamard", sf_no_fuse_hadamard),
            ("hopf_via_fusion", sf_hopf_via_fusion),
            ("triangle", sf_triangle),
            ("keep_hadamard_neighbor", sf_keep_hadamard_neighbor),
        ] + _random_cases(random_circuit, [11, 12]) + _random_cases(random_circuit, [13], clifford=True),
    },
    "identity": {
        "db_method": "remove_identities",
        "pyzx": pyzx_fixpoint(zx.id_simp),
        "cases": [
            ("simple_simple", id_simple_simple),
            ("chain", id_chain),
            ("hadamard_combo", id_hadamard_combo),
            ("double_hadamard", id_double_hadamard),
            ("x_identity", id_x_identity),
            ("nonzero_phase", id_nonzero_phase),
            ("in_triangle", id_in_triangle),
            ("in_triangle_hh", id_in_triangle_hh),
            ("in_triangle_ss", id_in_triangle_ss),
        ] + _random_cases(random_circuit, [21, 22]),
    },
    "supplementarity": {
        "db_method": "supplementarity_simp",
        "pyzx": pyzx_fixpoint(zx.supplementarity_simp),
        "cases": [
            ("basic_type1", supp_basic_type1),
            ("type2_adjacent", supp_type2_adjacent),
            ("type2_sum_zero", supp_type2_sum_zero),
            ("no_fire_clifford", supp_no_fire_clifford),
            ("no_fire_diff_neigh", supp_no_fire_diff_neigh),
            ("type1_diff_odd", supp_type1_diff_odd),
            ("type2_diff_odd", supp_type2_diff_odd),
            ("two_neighbor_pair", supp_two_neighbor_pair),
        ] + _random_cases(random_graphlike, [101, 102]),
    },
    "lcomp": {
        "db_method": "local_complementation_rule",
        "pyzx": pyzx_fixpoint(zx.lcomp_simp),
        "cases": [
            ("basic", lc_basic),
            ("toggle_off", lc_toggle_off),
            ("neg_phase", lc_neg_phase),
            ("three_neighbors", lc_three_neighbors),
            ("leaf_center", lc_leaf_center),
            ("no_match", lc_no_match),
            ("no_fire_boundary_center", lc_no_fire_boundary_center),
            ("no_fire_simple_edge", lc_no_fire_simple_edge),
            ("two_candidates_shared", lc_two_candidates_shared,
             {"require_iso": False}),
        ] + _random_cases(random_graphlike, [41], clifford=True) + _random_cases(random_graphlike, [42]),
    },
    "pivot": {
        "db_method": "pivot_rule",
        "pyzx": pyzx_fixpoint(zx.pivot_simp),
        "cases": [
            ("basic_00", pv_basic_00),
            ("mixed_01", pv_mixed_01),
            ("both_pi", pv_both_pi),
            ("shared_neighbor", pv_shared_neighbor),
            ("no_fire_nonpauli", pv_no_fire_nonpauli),
            ("no_fire_simple_edge", pv_no_fire_simple_edge),
            ("isolated_pair", pv_isolated_pair),
            ("boundary_deg2", pv_boundary_deg2),
            ("boundary_general", pv_boundary_general),
            ("no_fire_two_boundaries", pv_no_fire_two_boundaries),
        ] + _random_cases(random_graphlike, [51], clifford=True) + _random_cases(random_graphlike, [52]),
    },
    "pivot_gadget": {
        "db_method": "pivot_gadget_rule",
        "pyzx": pyzx_fixpoint(zx.pivot_gadget_simp),
        "cases": [
            ("basic", pg_basic),
            ("j_zero", pg_j_zero),
            ("shared_neighbor", pg_shared_neighbor),
            ("connected_neighbors", pg_connected_neighbors),
            ("no_fire_both_pauli", pg_no_fire_both_pauli),
            ("no_fire_tip", pg_no_fire_tip),
            ("no_fire_boundary", pg_no_fire_boundary),
        ] + _random_cases(random_graphlike, [61, 62]),
    },
    "pivot_boundary": {
        "db_method": "pivot_boundary_rule",
        "pyzx": pyzx_fixpoint(zx.pivot_boundary_simp),
        "cases": [
            ("basic", pb_basic),
            ("j_zero", pb_j_zero),
            ("w_nonclifford", pb_w_nonclifford),
            ("shared_neighbor", pb_shared_neighbor),
            ("no_fire_v_boundary", pb_no_fire_v_boundary),
        ] + _random_cases(random_graphlike, [71, 72]),
    },
    "gadget_fusion": {
        "db_method": "phase_gadget_fusion_rule",
        "pyzx": pyzx_fixpoint(zx.gadget_simp),
        "cases": [
            ("two_same_targets", gf_two_same_targets),
            ("axel_pi", gf_axel_pi),
            ("single_axel_pi", gf_single_axel_pi),
            ("three_gadgets", gf_three_gadgets),
            ("no_fuse_diff_targets", gf_no_fuse_diff_targets),
            ("no_fuse_clifford_leaf", gf_no_fuse_clifford_leaf),
            ("no_fuse_pauli_leaf", gf_no_fuse_pauli_leaf),
        ] + _random_cases(random_graphlike, [81, 82]),
    },
    "bialgebra": {
        "db_method": "bialgebra_simp",
        "pyzx": pyzx_fixpoint(zx.bialg_simp),
        "cases": [
            ("basic_k11", ba_basic_k11),
            ("k22", ba_k22),
            ("existing_edge", ba_existing_edge),
            ("pi_center", ba_pi_center),
            ("no_fire_nonpauli", ba_no_fire_nonpauli),
            ("no_fire_neighbor_phase", ba_no_fire_neighbor_phase),
            ("no_fire_wrong_color", ba_no_fire_wrong_color),
            ("no_fire_boundary_neighbor", ba_no_fire_boundary_neighbor),
        ] + _random_cases(random_circuit, [91, 92]),
    },
    "hadamard_cancel": {
        "db_method": "hadamard_cancel",
        "pyzx": pyzx_fixpoint(zx.id_simp),
        "cases": [
            ("single_pair", hc_single_pair),
            ("chain4", hc_chain4),
            ("chain3", hc_chain3, {"require_iso": False}),
            ("no_fire_phase", hc_no_fire_phase),
            ("boundary_ends", hc_boundary_ends),
        ] + _random_cases(random_circuit, [31, 32]),
    },
    "copy": {
        "db_method": "copy_simp",
        "pyzx": pyzx_fixpoint(pyzx_copy_simp),
        "cases": [
            ("a0_mixed", copy_a0_mixed),
            ("a1_interior", copy_a1_interior),
            ("only_boundaries", copy_only_boundaries),
            ("cascade", copy_cascade),
            ("two_leaves", copy_two_leaves),
            ("simple_edge_leaf", copy_simple_edge_leaf),
            ("x_leaf_simple", copy_x_leaf_simple),
            ("no_fire_h_diff_colors", copy_no_fire_h_diff_colors),
        ] + _random_cases(random_graphlike, [111, 112]),
    },
}
