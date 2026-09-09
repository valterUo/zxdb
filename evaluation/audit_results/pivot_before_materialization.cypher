// Interior pivot (pyzx match_pivot + pivot): two INTERIOR Pauli Z-spiders
// connected by a Hadamard edge. Interior = every wire is a Hadamard edge
// and every neighbor is a Z-spider. Toggles the complete bipartite edges
// between the exclusive/shared neighbor sets, copies b's phase onto a's
// exclusive neighbors and vice versa, adds a+b+1 to shared neighbors, then
// deletes both. ONE application per call; the driver loops.
//
// OPTIMIZED: interiorness of a and b is checked with cheap short-circuiting
// EXISTS predicates, and the O(degree) neighbour-set collection runs only
// for the single chosen pivot (after LIMIT 1) instead of for every
// Pauli-Pauli Hadamard edge in the graph.

MATCH (a:Node {graph_id: $graph_id, t: 1})-[pe:Wire {t: 2}]-(b:Node {t: 1})
WHERE id(a) < id(b)
  AND a.phase IS NOT NULL AND b.phase IS NOT NULL
  AND a.phase % 2 = round(a.phase % 2)
  AND b.phase % 2 = round(b.phase % 2)
  AND NOT EXISTS { MATCH (a)-[w:Wire]-(x:Node) WHERE w.t <> 2 OR x.t <> 1 }
  AND NOT EXISTS { MATCH (b)-[w:Wire]-(x:Node) WHERE w.t <> 2 OR x.t <> 1 }
WITH a, b
ORDER BY id(a), id(b)
LIMIT 1

// Collect neighbour groups for the chosen pivot only (both interior, so all
// wires are Hadamard).
OPTIONAL MATCH (a)-[:Wire {t: 2}]-(na:Node)
WHERE na <> b
WITH a, b, collect(DISTINCT na) AS a_nbrs
OPTIONAL MATCH (b)-[:Wire {t: 2}]-(nb:Node)
WHERE nb <> a
WITH a, b, a_nbrs, collect(DISTINCT nb) AS b_nbrs
WITH a, b,
     [x IN a_nbrs WHERE NOT x IN b_nbrs] AS only_a,
     [x IN b_nbrs WHERE NOT x IN a_nbrs] AS only_b,
     [x IN a_nbrs WHERE x IN b_nbrs] AS shared

CALL {
  WITH only_a, only_b
  UNWIND only_a AS u
  UNWIND only_b AS v
  OPTIONAL MATCH (u)-[existing:Wire]-(v)
  FOREACH (_ IN CASE WHEN existing IS NOT NULL THEN [1] ELSE [] END |
    DELETE existing)
  FOREACH (_ IN CASE WHEN existing IS NULL THEN [1] ELSE [] END |
    CREATE (u)-[:Wire {t: 2, graph_id: u.graph_id}]->(v))
}

CALL {
  WITH only_a, shared
  UNWIND only_a AS u
  UNWIND shared AS v
  OPTIONAL MATCH (u)-[existing:Wire]-(v)
  FOREACH (_ IN CASE WHEN existing IS NOT NULL THEN [1] ELSE [] END |
    DELETE existing)
  FOREACH (_ IN CASE WHEN existing IS NULL THEN [1] ELSE [] END |
    CREATE (u)-[:Wire {t: 2, graph_id: u.graph_id}]->(v))
}

CALL {
  WITH only_b, shared
  UNWIND only_b AS u
  UNWIND shared AS v
  OPTIONAL MATCH (u)-[existing:Wire]-(v)
  FOREACH (_ IN CASE WHEN existing IS NOT NULL THEN [1] ELSE [] END |
    DELETE existing)
  FOREACH (_ IN CASE WHEN existing IS NULL THEN [1] ELSE [] END |
    CREATE (u)-[:Wire {t: 2, graph_id: u.graph_id}]->(v))
}

FOREACH (n IN only_a |
  SET n.phase = (coalesce(n.phase, 0) + b.phase) % 2)
FOREACH (n IN only_b |
  SET n.phase = (coalesce(n.phase, 0) + a.phase) % 2)
FOREACH (n IN shared |
  SET n.phase = (coalesce(n.phase, 0) + a.phase + b.phase + 1) % 2)

DETACH DELETE a, b

RETURN count(*) AS pivot_operations_performed