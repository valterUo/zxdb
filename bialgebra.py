def bialgebra():

    """
    // 1. Match the pattern and gather neighbors (excluding the n1-n2 edge)
    MATCH (n1:Node {t:1})-[w:Wire {t:1}]-(n2:Node {t:2})

    // Gather n1's neighbors (except n2), and the relationship types
    OPTIONAL MATCH (n1)-[edge1]-(nb1)
    WHERE id(nb1) <> id(n2)
    WITH n1, n2, w, collect(nb1) AS n1_neighs, collect(edge1) AS n1_edges

    // Gather n2's neighbors (except n1), and the relationship types
    OPTIONAL MATCH (n2)-[edge2]-(nb2)
    WHERE id(nb2) <> id(n1)
    WITH n1, n2, w, n1_neighs, n1_edges, collect(nb2) AS n2_neighs, collect(edge2) AS n2_edges

    // 2. Prepare to multiply nodes
    WITH n1, n2, w, n1_neighs, n1_edges, n2_neighs, n2_edges,
        range(0, size(n1_neighs)-1) AS n1_idx,
        range(0, size(n2_neighs)-1) AS n2_idx

    // 3. Create new nodes for n1 (t:2)
    UNWIND n1_idx AS i1
    CREATE (new_n1:Node {t:2, uuid: randomUUID(), original: id(n1)})
    WITH n1, n2, w, n1_neighs, n1_edges, n2_neighs, n2_edges, collect(new_n1) AS new_n1s, n2_idx

    // 4. Create new nodes for n2 (t:1)
    UNWIND n2_idx AS i2
    CREATE (new_n2:Node {t:1, uuid: randomUUID(), original: id(n2)})
    WITH n1, n2, w, n1_neighs, n1_edges, n2_neighs, n2_edges, new_n1s, collect(new_n2) AS new_n2s

    // 5. Reconnect previous edges for new_n1 nodes (index-safe)
    WITH n1, n2, w, n1_neighs, n1_edges, n2_neighs, n2_edges, new_n1s, new_n2s
    UNWIND range(0, size(new_n1s)-1) AS i
    WITH n1_neighs[i] AS nb, n1_edges[i] AS edge, new_n1s[i] AS new_n1, n1, n2, w, n2_neighs, n2_edges, new_n2s
    CREATE (new_n1)-[:Wire {t: edge.t}]->(nb)
    WITH n1, n2, w, n2_neighs, n2_edges, new_n2s, collect(new_n1) AS new_n1s

    // 6. Reconnect previous edges for new_n2 nodes (index-safe)
    UNWIND range(0, size(new_n2s)-1) AS j
    WITH n2_neighs[j] AS nb, n2_edges[j] AS edge, new_n2s[j] AS new_n2, new_n1s, n1, n2
    CREATE (new_n2)-[:Wire {t: edge.t}]->(nb)
    WITH new_n1s, collect(new_n2) AS new_n2s, n1, n2

    // 7. Create all-to-all connections between the new nodes
    UNWIND new_n1s AS n1x
    UNWIND new_n2s AS n2x
    CREATE (n1x)-[:Wire {t:1}]->(n2x)

    // 8. Remove the original nodes and their connecting edge
    WITH n1, n2
    DETACH DELETE n1, n2
    """