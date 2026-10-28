# Adapted from PyZX 0.10.3's gadget-pivot matcher.
# Copyright (C) 2018 - Aleks Kissinger and John van de Wetering
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Equivalent gadget matching for certificate subprocesses, never timed PyZX.

The pinned PyZX matcher repeatedly searches a candidate list and copies incident
edges. An insertion-ordered dictionary preserves its pop/deletion order. During
matching, existing adjacency does not change: new edges are added only at the
end. Therefore adjacency, eligibility and incident edges can be cached per call.
Limited/explicit matches and other graph backends retain the original matcher.
"""
from collections import Counter

from pyzx.rewrite_rules import pivot_rule
from pyzx.utils import EdgeType, VertexType


_original_matcher = pivot_rule.match_pivot_gadget


def cached_gadget_match(g, vertices=None, num=-1):
    if g.backend != 'simple' or vertices is not None or num != -1:
        return _original_matcher(g, vertices=vertices, num=num)
    candidates = dict.fromkeys(Counter(g.edge_set()).elements())
    types, phases, rows = g.types(), g.phases(), g.rows()
    neighbors = {v: tuple(g.neighbors(v)) for v in g.vertices()}
    incident = {v: tuple(g.incident_edges(v)) for v in g.vertices()}
    interior = {v: all(types[n] == VertexType.Z for n in ns)
                for v, ns in neighbors.items()}
    leaf_neighbor = {v: any(len(incident[n]) == 1 for n in ns)
                     for v, ns in neighbors.items()}
    matches, new_edges = [], []
    while candidates:
        edge, _ = candidates.popitem()
        a, b = g.edge_st(edge)
        if types[a] != VertexType.Z or types[b] != VertexType.Z:
            continue
        if phases[a] not in (0, 1):
            if phases[b] not in (0, 1):
                continue
            a, b = b, a
        elif phases[b] in (0, 1):
            continue
        if g.is_ground(a) or len(neighbors[b]) == 1:
            continue
        if not interior[a] or not interior[b] or leaf_neighbor[a]:
            continue
        # The partner b has degree >= 2, so any degree-one neighbor of a
        # would be the forbidden gadget leaf, never the matched edge itself.
        discarded = set()
        for n in neighbors[a] + neighbors[b]:
            discarded.update(incident[n])
        tip = g.add_vertex(VertexType.Z, -2, rows[a], phases[b])
        g.set_phase(b, 0)
        g.set_qubit(a, -1)
        g.update_phase_index(b, tip)
        new_edges.append((tip, b))
        matches.append(((a, b), ([], [tip])))
        for candidate in discarded:
            candidates.pop(candidate, None)
    g.add_edges(new_edges, EdgeType.SIMPLE)
    return matches


def install_in_certificate_worker():
    """Call only inside the isolated identity-certificate subprocess."""
    pivot_rule.match_pivot_gadget = cached_gadget_match
