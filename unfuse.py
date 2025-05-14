def unfuse():

    """
    // 1. Match the fixed node and its properties
    MATCH (fixedNode:Node {id: 10})

    // 2. Create two new nodes, cloning properties
    CREATE (newNodeA:Node)
    SET newNodeA = fixedNode
    SET newNodeA.id = randomUUID()
    CREATE (newNodeB:Node)
    SET newNodeB = fixedNode
    SET newNodeB.id = randomUUID()

    // 3. Reconnect wires from SetA to newNodeA, copying edge properties
    WITH fixedNode, newNodeA, newNodeB
    UNWIND [1] AS aId
    MATCH (a:Node {id: aId})-[w:Wire]-(fixedNode)
    WITH a, w, newNodeA, fixedNode, newNodeB
    CREATE (a)-[newWire:Wire]->(newNodeA)
    SET newWire = w  // Copies all properties from w to newWire
    DELETE w

    // 4. Reconnect wires from SetB to newNodeB, copying edge properties
    WITH fixedNode, newNodeA, newNodeB
    UNWIND [12, 13] AS bId
    MATCH (b:Node {id: bId})-[w:Wire]-(fixedNode)
    WITH b, w, newNodeB, fixedNode, newNodeA
    CREATE (b)-[newWire:Wire]->(newNodeB)
    SET newWire = w  // Copies all properties from w to newWire
    DELETE w

    // 5. Create a :Wire edge between the two new nodes
    WITH newNodeA, newNodeB, fixedNode
    MERGE (newNodeA)-[:Wire {t : 1, graph_id : newNodeA.graph_id}]->(newNodeB)

    // 6. Remove the original fixed node
    WITH fixedNode
    DETACH DELETE fixedNode
    """