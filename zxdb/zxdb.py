import math
from neo4j import GraphDatabase
import json
import os
from typing import Optional
import logging

import numpy as np
import pyzx as zx

# Configure logging
logging.basicConfig(
    filename='app.log',            # Log file name
    filemode='w',                  # Append mode ('w' for overwrite)
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO             # Minimum logging level
)

class ZXdb:
    
    def __init__(self, neo4j_uri = "bolt://localhost:7687", neo4j_user = "", neo4j_password=""):
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.basic_rewrite_rule_queries = {}

        with open("zxdb/query_collections/memgraph-collection-zxdb.json", "r") as f:
            query_collection = json.load(f)
        
        for e in query_collection["items"]:
            self.basic_rewrite_rule_queries[e["title"]] = e



    def empty_graphdb(self, graph_id: str) -> None:
        """
        Clear all data from the graph database for a specific graph_id.
        
        Args:
            graph_id: Identifier for the graph to clear
        """
        driver = GraphDatabase.driver(self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password))
        
        try:
            with driver.session() as session:
                def clear_graph(tx):
                    tx.run("""
                        MATCH (v:Node {graph_id: $graph_id})
                        DETACH DELETE v
                    """, graph_id=graph_id)
                    
                    return True
                
                session.execute_write(clear_graph)
                logging.info(f"Cleared graph with ID '{graph_id}'")
        finally:
            # Close the driver connection
            driver.close()

    
    def export_graphdb_to_zx_graph(self,
        graph_id: str,
        json_file_path: str
        ) -> None:
        """        
        Export a graph from Neo4j or Memgraph database to PyZX graph.
        Args:
            graph_id: Identifier for the graph to export
            json_file_path: Path to save the exported JSON file
        The function retrieves:
        - Vertex nodes with properties
        - Relationships between vertices based on edges
        """
        driver = GraphDatabase.driver(self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password))
        g = zx.Graph()

        try:
            with driver.session() as session:

                def fetch_graph_data(tx):
                    # Fetch vertices and edges
                    vertices_query = """
                        MATCH (v:Node {graph_id: $graph_id})
                        RETURN v.id AS id, v.t AS t, v.phase AS phase, v AS v
                    """
                    edges_query = """
                        MATCH (source:Node {graph_id: $graph_id})-[r:Wire]->(target:Node {graph_id: $graph_id})
                        RETURN source.id AS source_id, target.id AS target_id, r.t AS t
                    """
                    input_vertices_query = """
                        MATCH (v:Node:Input {graph_id: $graph_id})
                        RETURN v.id AS id
                    """
                    output_vertices_query = """
                        MATCH (v:Node:Output {graph_id: $graph_id})
                        RETURN v.id AS id
                    """
                    
                    vertices = tx.run(vertices_query, graph_id=graph_id).data()
                    edges = tx.run(edges_query, graph_id=graph_id).data()
                    
                    return vertices, edges
                
                vertices, edges = session.execute_read(fetch_graph_data)

                # Add vertices to the graph
                vertex_ids = {}
                for vertex in vertices:
                    vid = g.add_vertex(ty=vertex['t'], phase=vertex.get('phase', None))
                    vertex_ids[vertex['id']] = vid
                
                # Add edges to the graph
                for edge in edges:
                    g.add_edge((vertex_ids[edge['source_id']], vertex_ids[edge['target_id']]), edgetype=edge['t'])

                # Write the graph data to a JSON file
                graph_data = g.to_json()
                with open(json_file_path, 'w') as f:
                    json.dump(graph_data, f, indent=2)
                
                logging.info(f"Graph data exported to {json_file_path}")
        finally:
            # Close the driver connection
            driver.close()
        
        return g


    def import_zx_graph_json_to_graphdb(self,
        json_file_path: str,
        graph_id: Optional[str] = None,
        save_metadata: bool = True,
        initialize_empty: bool = False,
        batch_size: int = 5000
        ) -> None:
        """
        Import a graph JSON file into Neo4j or Memgraph database, storing only vertices and edges.
        Uses efficient batch operations for faster imports.
        
        Args:
            json_file_path: Path to the JSON file containing graph data
            neo4j_uri: URI for the Neo4j/Memgraph instance
            neo4j_user: Database username
            neo4j_password: Database password
            graph_id: Optional identifier for the graph (uses filename if not provided)
            save_metadata: Whether to save metadata to a separate file
            initialize_empty: Whether to clear existing graph data before import
            batch_size: Number of elements to include in each batch operation
        
        The function creates:
        - Vertex nodes with properties
        - Relationships between vertices based on edges
        """
        # Load JSON data
        with open(json_file_path, 'r') as f:
            graph_data = json.load(f)
        
        # Extract graph ID from file path if not provided
        if graph_id is None:
            graph_id = os.path.basename(json_file_path).split('.')[0]
        
        # Save metadata to separate file if requested
        if save_metadata:
            metadata = {
                "graph_id": graph_id,
                "version": graph_data.get("version"),
                "backend": graph_data.get("backend"),
                "variable_types": graph_data.get("variable_types", {}),
                "scalar": graph_data.get("scalar", {}),
                "vertices": len(graph_data.get("vertices", [])),
                "edges": len(graph_data.get("edges", [])),
                "inputs": len(graph_data.get("inputs", [])),
                "outputs": len(graph_data.get("outputs", []))
            }
            
            metadata_file = f"{os.path.splitext(json_file_path)[0]}_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            logging.info(f"Metadata saved to {metadata_file}")
            
        with GraphDatabase.driver(self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password)).session(database="memgraph") as session:
            session.run("CREATE INDEX ON :Node(id);")
        
        driver = GraphDatabase.driver(self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password))
        
        try:
            # Clear existing graph data if requested
            if initialize_empty:
                with driver.session() as session:
                    def clear_existing_graph(tx):
                        tx.run("""
                            MATCH (v)
                            DETACH DELETE v
                        """, graph_id=graph_id)
                        
                        return True
                    
                    session.execute_write(clear_existing_graph)
                    
                    logging.info(f"Cleared existing graph with ID '{graph_id}'")
        
            with driver.session() as session:
                def create_vertices(tx):
                    # Prepare and process vertices in batches
                    vertices = graph_data.get("vertices", [])
                    for i in range(0, len(vertices), batch_size):
                        batch = vertices[i:i+batch_size]
                        vertices_batch = []
                        
                        for vertex in batch:
                            # Create vertex properties dictionary
                            vertex_props = {
                                "graph_id": graph_id,
                                "id": vertex["id"],
                                "t": vertex.get("t")
                            }
                            
                            if "phase" in vertex:
                                pi_string = "\u03c0"
                                if isinstance(vertex["phase"], str) and pi_string in vertex["phase"]:
                                    try:
                                        expression = vertex["phase"].replace("~", "")
                                        expression = expression.replace(pi_string, f"*{math.pi}")
                                        # Fix cases like 'π' at the start (e.g., 'π/2')
                                        if expression.startswith(f"*{math.pi}"):
                                            expression = expression.replace(f"*{math.pi}", str(math.pi), 1)
                                        numeric_value = eval(expression)
                                        vertex_props["phase"] = float(numeric_value)
                                    except Exception as e:
                                        print(f"Error evaluating phase expression '{vertex['phase']}': {e}")
                                elif isinstance(vertex["phase"], (int, float)):
                                    # Keep phase as is if it's already a number
                                    vertex_props["phase"] = float(vertex["phase"])
                            else:
                                if vertex.get("t") != 0:
                                    # Default to pi if phase is not a valid number or string
                                    vertex_props["phase"] = math.pi
                            
                            # Add any additional properties
                            for k, v in vertex.items():
                                if k not in ["id", "t", "pos", "phase"]:
                                    vertex_props[k] = v
                            
                            vertices_batch.append(vertex_props)
                        
                        # Batch create vertices
                        if vertices_batch:
                            tx.run("""
                                UNWIND $vertices AS vertex
                                CREATE (v:Node)
                                SET v = vertex
                            """, vertices=vertices_batch)
                            
                        logging.info(f"Vertex batch {i} of {np.ceil(len(vertices) / batch_size)} stored.")
                    
                    # Batch mark input vertices with Input label in batches
                    if "inputs" in graph_data and graph_data["inputs"]:
                        inputs = graph_data["inputs"]
                        for i in range(0, len(inputs), batch_size):
                            batch = inputs[i:i+batch_size]
                            tx.run("""
                                UNWIND $input_ids AS input_id
                                MATCH (v:Node {graph_id: $graph_id, id: input_id})
                                SET v:Input
                            """, graph_id=graph_id, input_ids=batch)
                    
                    # Batch mark output vertices with Output label in batches
                    if "outputs" in graph_data and graph_data["outputs"]:
                        outputs = graph_data["outputs"]
                        for i in range(0, len(outputs), batch_size):
                            batch = outputs[i:i+batch_size]
                            tx.run("""
                                UNWIND $output_ids AS output_id
                                MATCH (v:Node {graph_id: $graph_id, id: output_id})
                                SET v:Output
                            """, graph_id=graph_id, output_ids=batch)
                            
                session.execute_write(create_vertices)
                
                        
            with driver.session() as session:
                def create_edges(tx):
                    # Prepare and process edges in batches
                    edges = graph_data.get("edges", [])
                    for i in range(0, len(edges), batch_size):
                        batch = edges[i:i+batch_size]
                        edges_batch = []
                        
                        for edge in batch:
                            # Edge format is [source_id, target_id, type]
                            if len(edge) >= 3:
                                edges_batch.append({
                                    "source_id": edge[0],
                                    "target_id": edge[1],
                                    "t": edge[2],
                                    "graph_id": graph_id
                                })
                        
                        # Batch create edges
                        if edges_batch:
                            tx.run("""
                                UNWIND $edges AS edge
                                MATCH (source:Node {graph_id: $graph_id, id: edge.source_id})
                                MATCH (target:Node {graph_id: $graph_id, id: edge.target_id})
                                CREATE (source)-[r:Wire {
                                    t: edge.t,
                                    graph_id: edge.graph_id
                                }]->(target)
                            """, edges=edges_batch, graph_id=graph_id)

                        logging.info(f"Edge batch {i} of {np.ceil(len(edges) / batch_size)} stored.")
                        
                session.execute_write(create_edges)
        finally:
            driver.close()


    def hadamard_cancel(self, graph_id: str) -> None:
        """
        Cancel Hadamard gates in the graph by removing pairs of Hadamard gates on the same qubit.
        
        Args:
            graph_id: Identifier for the graph to process
        """
        driver = GraphDatabase.driver(self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password))
        
        try:
            with driver.session() as session:

                def cancel_hadamards(tx):

                    query = str(self.basic_rewrite_rule_queries["Cancel Hadamard gates"]["query"]["code"]["value"])
                    tx.run(query, graph_id=graph_id)

                    query = str(self.basic_rewrite_rule_queries["Cancel Hadamard patterns"]["query"]["code"]["value"])
                    tx.run(query, graph_id=graph_id)

                    logging.info(f"Hadamard cancellation completed for graph ID '{graph_id}'")
                
                session.execute_write(cancel_hadamards)
        finally:
            driver.close()


    def remove_identities(self, graph_id: str) -> None:
        """
        Remove identity gates from the graph.
        
        Args:
            graph_id: Identifier for the graph to process
        """
        driver = GraphDatabase.driver(self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password))
        
        try:
            with driver.session() as session:

                def removes_ids(tx):

                    query = str(self.basic_rewrite_rule_queries["Remove identities"]["query"]["code"]["value"])
                    tx.run(query, graph_id=graph_id)

                    logging.info(f"Hadamard cancellation completed for graph ID '{graph_id}'")
                
                session.execute_write(removes_ids)
        finally:
            driver.close()