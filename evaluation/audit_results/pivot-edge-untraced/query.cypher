// pyzx full_reduce ends with remove_isolated_vertices(): degree-0 nodes
// only contribute a scalar and are dropped.

MATCH (n:Node {graph_id: $graph_id})
WHERE degree(n) = 0
DETACH DELETE n
RETURN count(*) AS removed