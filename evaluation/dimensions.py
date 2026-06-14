"""
Which circuit dimension drives the DB-vs-pyzx full_reduce crossover?

A controlled study. One parametric generator with INDEPENDENT knobs is swept one
factor at a time (OFAT) around a fixed baseline, and for every circuit we record
both the timing (DB full_reduce vs pyzx, round-trips) and INTRINSIC graph metrics
so the crossover is explained by a measured property, not a circuit label:

  knobs            qubits (WIDTH), layers (DEPTH), degree (DENSITY),
                   span (LOCALITY, max |i-j| of an interaction; None = full),
                   phase (GADGET content: 1/2 Clifford .. 1/4 T)
  intrinsic metrics  input vertices, max/mean spider degree,
                   tw_in  = heuristic treewidth of the input circuit graph,
                   tw_red = heuristic treewidth of the (pyzx) reduced graph
                            -- the irreducible core both engines work on.

Width is run at BOTH a fixed local span (treewidth bounded -> isolates pure
width) and full non-local span, to settle whether width itself matters or only
the treewidth it can induce. The pooled tail prints every run sorted by tw_red so
the band of treewidth where the DB wins (ratio < 1) is visible directly.

Uses the production full_reduce. Usage:
    python -m evaluation.dimensions verify
    python -m evaluation.dimensions width_local locality ...   # selected
    python -m evaluation.dimensions                            # all
"""
import io
import random
import subprocess
import sys
import time
from contextlib import redirect_stdout
from fractions import Fraction

import networkx as nx
from networkx.algorithms.approximation import treewidth_min_degree
import pyzx as zx

from zxdb.zxdb import ZXdb
from utils import zx_graph_to_db, pyzx_fixpoint
from evaluation.harness import verify_pair
from evaluation.benchmark_perf import _count_round_trips

GID = "example_graph"
PR = pyzx_fixpoint(lambda h: zx.full_reduce(h, quiet=True))
TIME_CAP = 12.0

# baseline: moderate width/depth, sparse, non-local, T-phase (near the QAOA
# crossover regime). Individual sweeps override knobs as needed (below).
BASE = dict(qubits=28, layers=2, degree=3, span=None, phase=Fraction(1, 4))


def _build_edges(qubits, degree, span, rng):
    if span is None:
        span = qubits
    edges = set()
    target = max(1, degree * qubits // 2)
    attempts = 0
    while len(edges) < target and attempts < target * 40:
        a = rng.randrange(qubits)
        lo = max(0, a - span); hi = min(qubits - 1, a + span)
        b = rng.randrange(lo, hi + 1)
        if a != b:
            edges.add((min(a, b), max(a, b)))
        attempts += 1
    return edges


def _layer_edges(qubits, layers, degree, span, seed):
    """The per-layer interaction edge sets (deterministic for a seed)."""
    rng = random.Random(seed)
    return [_build_edges(qubits, degree, span, rng) for _ in range(layers)]


def gen_from(qubits, layer_edges, phase):
    c = zx.Circuit(qubits)
    for q in range(qubits):
        c.add_gate("HAD", q)
    for edges in layer_edges:
        for (a, b) in edges:
            c.add_gate("CNOT", a, b)
            c.add_gate("ZPhase", b, phase=phase)
            c.add_gate("CNOT", a, b)
        for q in range(qubits):
            c.add_gate("XPhase", q, phase=Fraction(1, 2))
    return c.to_graph()


def gen(qubits, layers, degree, span, phase, seed):
    """Layered interaction ansatz: each layer applies ZZ(phase) on a random
    degree-`degree`, range-`span` interaction graph (a phase gadget when phase
    is non-Clifford), then an X(pi/2) mixer. Generalizes QAOA / Trotter."""
    return gen_from(qubits, _layer_edges(qubits, layers, degree, span, seed),
                    phase)


def _interaction_stats(qubits, layers, degree, span, seed):
    """Treewidth and edge count of the UNION interaction graph (the abstract
    qubit-coupling graph, independent of the ZX gate spiders) — the cleanest
    structural measure of the circuit's connectivity / locality."""
    edges = set().union(*_layer_edges(qubits, layers, degree, span, seed)) \
        if layers else set()
    H = nx.Graph(); H.add_nodes_from(range(qubits)); H.add_edges_from(edges)
    tw = treewidth_min_degree(H)[0] if H.number_of_edges() else 0
    return tw, H.number_of_edges()


def _nx(g):
    H = nx.Graph()
    H.add_nodes_from(g.vertices())
    for e in g.edges():
        s, t = g.edge_st(e)
        if s != t:
            H.add_edge(s, t)
    return H


def _treewidth(g):
    H = _nx(g)
    if H.number_of_nodes() == 0:
        return 0
    return treewidth_min_degree(H)[0]


def _deg_stats(g):
    degs = [len(list(g.neighbors(v))) for v in g.vertices()
            if g.type(v) != zx.VertexType.BOUNDARY]
    if not degs:
        return 0, 0.0
    return max(degs), sum(degs) / len(degs)


def _load(z, g):
    with redirect_stdout(io.StringIO()):
        zx_graph_to_db(z, g.copy(), graph_id=GID)


NSEEDS = 3   # circuits per point; metrics are averaged to suppress seed noise


def _recover(z):
    """Memgraph on this Win/Docker setup crashes under sustained full_reduce
    load; restart it and reconnect so a long sweep survives a crash."""
    try:
        z.close()
    except Exception:
        pass
    z._driver = None
    subprocess.run("docker restart memgraph", shell=True, capture_output=True)
    for _ in range(40):
        try:
            z.wipe_database()
            return True
        except Exception:
            time.sleep(2)
    return False


def _measure_one(z, rt, cfg, seed):
    g = gen(seed=seed, **cfg)
    nin = g.num_vertices()
    maxd, _ = _deg_stats(g)
    tw_in = _treewidth(g)
    tw_int, n_ie = _interaction_stats(cfg["qubits"], cfg["layers"],
                                      cfg["degree"], cfg["span"], seed)
    pg = g.copy()
    t = time.perf_counter(); PR(pg); t_pyzx = time.perf_counter() - t
    tw_red = _treewidth(pg)
    z.wipe_database(); _load(z, g)
    z.full_reduce(GID)               # warm
    z.wipe_database(); _load(z, g)
    rt["n"] = 0
    t = time.perf_counter(); z.full_reduce(GID); t_db = time.perf_counter() - t
    return dict(nin=nin, maxd=maxd, tw_in=tw_in, tw_int=tw_int, n_ie=n_ie,
                tw_red=tw_red, rt=rt["n"], t_db=t_db, t_pyzx=t_pyzx,
                ratio=t_db / max(t_pyzx, 1e-9))


def measure(z, rt, cfg, seed):
    """Average over NSEEDS circuits; also report how many seeds the DB won."""
    runs = [_measure_one(z, rt, cfg, seed + 1000 * k) for k in range(NSEEDS)]
    keys = ["nin", "maxd", "tw_in", "tw_int", "n_ie", "tw_red", "rt",
            "t_db", "t_pyzx", "ratio"]
    out = {k: sum(r[k] for r in runs) / len(runs) for k in keys}
    for k in ("nin", "maxd", "tw_in", "tw_int", "n_ie", "tw_red", "rt"):
        out[k] = round(out[k])
    out["db_wins"] = sum(1 for r in runs if r["ratio"] < 1)
    return out


# sweep -> (knob, values, base-overrides)
SWEEPS = {
    # WIDTH at fixed LOCAL span -> treewidth bounded -> isolates pure width
    "width_local":    ("qubits", [12, 18, 24, 30, 36, 44], dict(span=3)),
    # WIDTH at full non-local span -> treewidth grows with width
    "width_nonlocal": ("qubits", [12, 18, 24, 30, 36], dict(span=None)),
    # DEPTH at fixed local span -> isolates depth
    "depth":          ("layers", [1, 2, 3, 4, 6, 8], dict(span=3)),
    "density":        ("degree", [1, 2, 3, 5], dict()),
    # LOCALITY: the suspected driver. vary interaction range at fixed width
    "locality":       ("span", [1, 2, 4, 8, 16, None], dict(qubits=28, degree=3)),
    "phase":          ("phase", [Fraction(1, 2), Fraction(1, 4), Fraction(1, 8)],
                       dict()),
}


def sweep(z, rt, name):
    knob, values, over = SWEEPS[name]
    base = dict(BASE); base.update(over)
    print(f"\n### {name}   knob={knob}   (base {base})")
    print(f"{knob:>8} {'inV':>5} {'iE':>4} {'twInt':>5} {'twIn':>4} {'twRed':>5} "
          f"{'rt':>4} | {'db':>7} {'pyzx':>7} | ratio  win(n/{NSEEDS})")
    rows = []
    for i, val in enumerate(values):
        cfg = dict(base); cfg[knob] = val
        try:
            m = measure(z, rt, cfg, seed=1000 + 17 * i)
        except Exception as e:
            # crash on this Win/Docker Memgraph -> restart and retry once
            if "connect" in str(e).lower() or "refused" in str(e).lower() \
                    or "deleted node" in str(e).lower():
                print(f"{str(val):>8}  (memgraph crash; recovering...)", flush=True)
                if not _recover(z):
                    print("  recovery failed, stopping"); break
                try:
                    m = measure(z, rt, cfg, seed=1000 + 17 * i)
                except Exception as e2:
                    print(f"{str(val):>8}  FAIL after recover {str(e2)[:40]}")
                    continue
            else:
                print(f"{str(val):>8}  FAIL {str(e)[:42]}")
                break
        m["knob"] = name; m["val"] = (cfg["span"] if knob == "span" else val)
        rows.append(m)
        win = "DB " if m["ratio"] < 1 else "pyz"
        print(f"{str(val):>8} {m['nin']:>5} {m['n_ie']:>4} {m['tw_int']:>5} "
              f"{m['tw_in']:>4} {m['tw_red']:>5} {m['rt']:>4} | "
              f"{m['t_db']:>6.2f}s {m['t_pyzx']:>6.2f}s | "
              f"{m['ratio']:>5.2f}x {win} {m['db_wins']}/{NSEEDS}", flush=True)
        if max(m["t_db"], m["t_pyzx"]) > TIME_CAP:
            print("         (time cap)"); break
    return rows


def verify():
    """Tensor-verify the generator at small sizes across dimension extremes."""
    z = ZXdb(); rt = _count_round_trips()
    cfgs = [dict(qubits=4, layers=1, degree=3, span=None, phase=Fraction(1, 4)),
            dict(qubits=6, layers=1, degree=2, span=1, phase=Fraction(1, 4)),
            dict(qubits=6, layers=1, degree=5, span=None, phase=Fraction(1, 4)),
            dict(qubits=5, layers=2, degree=3, span=None, phase=Fraction(1, 2)),
            dict(qubits=5, layers=2, degree=3, span=None, phase=Fraction(1, 8))]
    print("verifying generator (tensor-feasible):")
    try:
        for i, cfg in enumerate(cfgs):
            g = gen(seed=7000 + i, **cfg); orig = g.copy()
            pg = g.copy(); PR(pg)
            z.wipe_database(); _load(z, g); z.full_reduce(GID)
            db = z.export_graphdb_to_zx_graph(GID, "v.json")
            ok, lvl = verify_pair(db, pg, original=orig)
            print(f"  {cfg}: {'OK' if ok else 'FAIL'} [{lvl}]", flush=True)
    finally:
        rt["restore"](); z.wipe_database(); z.close()


def _analysis(all_rows):
    print("\n" + "=" * 72)
    print("SENSITIVITY (ratio range per swept dimension; <1 => DB wins):")
    by = {}
    for r in all_rows:
        by.setdefault(r["knob"], []).append(r)
    for name, rows in by.items():
        rr = [r["ratio"] for r in rows]
        tw = [r["tw_red"] for r in rows]
        cross = "  CROSSES 1x" if min(rr) < 1 <= max(rr) else (
            "  DB wins throughout" if max(rr) < 1 else "")
        print(f"  {name:<15} ratio {min(rr):.2f}..{max(rr):.2f}  "
              f"twRed {min(tw)}..{max(tw)}{cross}")
    print("\nPOOLED: every run sorted by INTERACTION-GRAPH treewidth (twInt) ->")
    print("if treewidth is the driver, the DB-win band is a contiguous twInt range:")
    print(f"{'twInt':>5} {'iE':>4} {'twRed':>5} {'inV':>5} {'ratio':>6}  win  source")
    for r in sorted(all_rows, key=lambda x: (x["tw_int"], x["ratio"])):
        win = "DB " if r["ratio"] < 1 else "pyz"
        print(f"{r['tw_int']:>5} {r['n_ie']:>4} {r['tw_red']:>5} {r['nin']:>5} "
              f"{r['ratio']:>5.2f}x {win}  {r['knob']}")


def _warmup(z):
    """The memgraph-mage image loads PyTorch/DGL for ~60s after start, and a
    cold query plan cache makes the first reductions 5-15x slower and noisy.
    Run a few throwaway full_reduces until the time stabilizes so the measured
    sweep is not polluted by startup/ML-load contention. (With the autocommit
    fix there are no mid-run crashes, so the whole study runs restart-free.)"""
    prev = None
    for i in range(8):
        g = gen(qubits=20, layers=2, degree=3, span=None, phase=Fraction(1, 4),
                seed=900 + i)
        z.wipe_database(); _load(z, g)
        t = time.perf_counter(); z.full_reduce(GID); dt = time.perf_counter() - t
        if prev is not None and dt < 1.5 * prev and dt < 2.0:
            break
        prev = dt
    print(f"warmed up (last throwaway full_reduce {dt:.2f}s)", flush=True)


def main(names):
    z = ZXdb(); rt = _count_round_trips()
    _warmup(z)
    all_rows = []
    try:
        for name in names:
            all_rows += sweep(z, rt, name)
        _analysis(all_rows)
    finally:
        rt["restore"](); z.wipe_database(); z.close()


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if args == ["verify"]:
        verify()
    else:
        names = args or list(SWEEPS.keys())
        bad = [n for n in names if n not in SWEEPS]
        if bad:
            print(f"Unknown: {bad}. Available: {list(SWEEPS.keys())}")
            sys.exit(2)
        main(names)
