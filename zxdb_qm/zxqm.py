"""In-process pivot_gadget for zxdb, ported from pyzx's match_pivot_gadget + pivot.

Runs the whole pivot_gadget fixpoint inside Memgraph (one CALL, no round-trips):
each pass finds a MAXIMAL set of gadget matches with pairwise-disjoint closed
neighbourhoods (pyzx's greedy: mark a match's neighbourhood consumed, skip
overlapping candidates), applies them all, and repeats until none remain. This
is the imperative O(E) greedy that declarative Cypher / MAGE coloring cannot do
cheaply (the explicit conflict graph is quadratic on dense inputs).

Semantics mirror the Cypher "Pivot gadget" rule exactly:
  interior Pauli Z-spider z_j -H- interior non-Pauli Z-spider z_alpha (deg>1),
  z_j not a phase-gadget axis. Toggle complete bipartite edges among the
  exclusive/shared neighbour groups; only_alpha += z_j.phase; shared += z_j.phase
  + 1; create axis(phase z_j) -H- tip(phase z_alpha), axis inherits z_j's
  exclusive+shared neighbours; delete z_j and z_alpha.
"""
import mgp

_EPS = 1e-9
_WIRE = mgp.EdgeType("Wire")


def _ph(v):
    p = v.properties.get("phase")
    return 0.0 if p is None else float(p)


def _is_pauli(p):
    r = p % 2.0
    return abs(r) < _EPS or abs(r - 1.0) < _EPS or abs(r - 2.0) < _EPS


def _is_nonpauli(p):
    return not _is_pauli(p)


def _t(v):
    return v.properties.get("t")


def _nbr_map(v):
    """neighbour_id -> (edge, neighbour_vertex), both directions."""
    m = {}
    for e in v.out_edges:
        m[e.to_vertex.id] = (e, e.to_vertex)
    for e in v.in_edges:
        m[e.from_vertex.id] = (e, e.from_vertex)
    return m


def _degree(v):
    return sum(1 for _ in v.out_edges) + sum(1 for _ in v.in_edges)


def _interior(v, nbr_map):
    """all incident wires Hadamard (t==2) and all neighbours Z (t==1)."""
    for (e, n) in nbr_map.values():
        if e.properties.get("t") != 2 or n.properties.get("t") != 1:
            return False
    return True


def _add_h(g, u, w, gid):
    e = g.create_edge(u, w, _WIRE)
    e.properties.set("t", 2)
    e.properties.set("graph_id", gid)


def _toggle(g, A, B, gid):
    """toggle the complete bipartite Hadamard edges between vertex lists A, B."""
    for a in A:
        anb = {}
        for e in a.out_edges:
            anb[e.to_vertex.id] = e
        for e in a.in_edges:
            anb[e.from_vertex.id] = e
        for b in B:
            ex = anb.get(b.id)
            if ex is not None:
                g.delete_edge(ex)
            else:
                _add_h(g, a, b, gid)


def _apply(g, zj, za, zjn, zan):
    gid = zj.properties.get("graph_id")
    pj, pa = _ph(zj), _ph(za)
    jset = set(zjn.keys()) - {za.id}
    aset = set(zan.keys()) - {zj.id}
    only_j = jset - aset
    only_a = aset - jset
    shared = jset & aset
    vmap = {}
    for vid in (only_j | only_a | shared):
        vmap[vid] = zjn[vid][1] if vid in zjn else zan[vid][1]
    Lj = [vmap[i] for i in only_j]
    La = [vmap[i] for i in only_a]
    Ls = [vmap[i] for i in shared]
    _toggle(g, Lj, La, gid)
    _toggle(g, Lj, Ls, gid)
    _toggle(g, La, Ls, gid)
    for v in La:
        v.properties.set("phase", (_ph(v) + pj) % 2)
    for v in Ls:
        v.properties.set("phase", (_ph(v) + pj + 1) % 2)
    axis = g.create_vertex()
    axis.add_label("Node")
    axis.properties.set("t", 1)
    axis.properties.set("phase", pj % 2)
    axis.properties.set("graph_id", gid)
    tip = g.create_vertex()
    tip.add_label("Node")
    tip.properties.set("t", 1)
    tip.properties.set("phase", ((pa % 2) + 2) % 2)
    tip.properties.set("graph_id", gid)
    _add_h(g, axis, tip, gid)
    for v in Lj:
        _add_h(g, axis, v, gid)
    for v in Ls:
        _add_h(g, axis, v, gid)
    g.detach_delete_vertex(zj)
    g.detach_delete_vertex(za)


def _one_pass(g):
    consumed = set()
    applied = 0
    for zj in sorted(g.vertices, key=lambda x: x.id):
        if not zj.is_valid() or zj.id in consumed:
            continue
        if _t(zj) != 1:
            continue
        pj = _ph(zj)
        if not _is_pauli(pj):
            continue
        zjn = _nbr_map(zj)
        if not _interior(zj, zjn):
            continue
        if any(_degree(n) == 1 for (_, n) in zjn.values()):  # z_j is a gadget axis
            continue
        chosen = None
        for vid in sorted(zjn.keys()):
            e, za = zjn[vid]
            if za.id in consumed or _t(za) != 1:
                continue
            if not _is_nonpauli(_ph(za)) or _degree(za) <= 1:
                continue
            zan = _nbr_map(za)
            if not _interior(za, zan):
                continue
            claim = {zj.id, za.id} | set(zjn.keys()) | set(zan.keys())
            if claim & consumed:
                continue
            chosen = (za, zan, claim)
            break
        if chosen is None:
            continue
        za, zan, claim = chosen
        _apply(g, zj, za, zjn, zan)
        consumed |= claim
        applied += 1
    return applied


@mgp.write_proc
def pivot_gadget_fixpoint(ctx: mgp.ProcCtx,
                          graph_id: str = "") -> mgp.Record(applied=int):
    g = ctx.graph
    total = 0
    while True:
        n = _one_pass(g)
        total += n
        if n == 0:
            break
    return mgp.Record(applied=total)
