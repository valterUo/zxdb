from fractions import Fraction
from neo4j import GraphDatabase
import json
import os
from typing import Optional
import logging
import time

import numpy as np
import pyzx as zx
from pyzx.graph import VertexType

from zxdb.pyzx_utils import pi_string_to_fraction
import networkx as nx

# Configure logging
logging.basicConfig(
    filename='app.log',            # Log file name
    filemode='w',                  # Append mode ('w' for overwrite)
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO             # Minimum logging level
)

class ZXdb:
    
    def __init__(self, uri = "bolt://localhost:7687", user = "", password=""):
        self.uri = uri
        self.user = user
        self.password = password
        self.basic_rewrite_rule_queries = {}
        self._driver = None

        with open("zxdb/query_collections/memgraph-collection-zxdb.json", "r") as f:
            query_collection = json.load(f)
        
        for e in query_collection["items"]:
            self.basic_rewrite_rule_queries[e["title"]] = e

        with open("zxdb/query_collections/collection-Rewrite-queries-ZXdb.json", "r") as f:
            query_collection = json.load(f)
        
        for e in query_collection["items"]:
            self.basic_rewrite_rule_queries[e["title"]] = e
        
        with open("zxdb/query_collections/collection-Labeling-queries-ZXdb.json", "r") as f:
            query_collection = json.load(f)

        for e in query_collection["items"]:
            self.basic_rewrite_rule_queries[e["title"]] = e
    
    @property
    def driver(self):
        """Create driver only when needed"""
        if self._driver is None:
            self._driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        return self._driver
    
    def close(self):
        """Explicitly close the driver"""
        if self._driver is not None:
            self._driver.close()
            self._driver = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


    def empty_graphdb(self, graph_id: str) -> None:
        """
        Clear all data from the graph database for a specific graph_id.
        
        Args:
            graph_id: Identifier for the graph to clear
        """
    
        with self.driver.session() as session:

            def clear_graph(tx):
                tx.run("""
                    MATCH (v:Node {graph_id: $graph_id})
                    DETACH DELETE v
                """, graph_id=graph_id)
                
                return True
            
            session.execute_write(clear_graph)
            logging.info(f"Cleared graph with ID '{graph_id}'")

    
    def export_graphdb_to_zx_graph(self,
        graph_id: str,
        json_file_path: str
        ) -> zx.Graph:
        """
        Export a graph from Neo4j or Memgraph database to a PyZX graph and write JSON.
        Positions are computed (spring layout) and stored so they appear as 'pos' in JSON.
        """
        g = zx.Graph(backend="multigraph")

        with self.driver.session() as session:

            def fetch_graph_data(tx):
                vertices_query = """
                    MATCH (v:Node {graph_id: $graph_id})
                    RETURN id(v) AS id, v.t AS t, v.phase AS phase
                """
                edges_query = """
                    MATCH (source:Node {graph_id: $graph_id})-[r:Wire]->(target:Node {graph_id: $graph_id})
                    RETURN id(source) AS source_id, id(target) AS target_id, r.t AS t
                """
                input_vertices_query = """
                    MATCH (v:Node:Input {graph_id: $graph_id})
                    RETURN id(v) AS id
                """
                output_vertices_query = """
                    MATCH (v:Node:Output {graph_id: $graph_id})
                    RETURN id(v) AS id
                """

                vertices = tx.run(vertices_query, graph_id=graph_id).data()
                edges = tx.run(edges_query, graph_id=graph_id).data()
                inputs = tx.run(input_vertices_query, graph_id=graph_id).data()
                outputs = tx.run(output_vertices_query, graph_id=graph_id).data()
                return vertices, edges, inputs, outputs

            vertices, edges, inputs, outputs = session.execute_read(fetch_graph_data)

            # Add vertices
            vertex_ids = {}
            for vertex in vertices:
                t = vertex['t']
                if t == 0:
                    vtype = VertexType.BOUNDARY
                elif t == 1:
                    vtype = VertexType.Z
                elif t == 2:
                    vtype = VertexType.X
                elif t == 3:
                    vtype = VertexType.H_BOX
                elif t == 4:
                    vtype = VertexType.W_INPUT
                elif t == 5:
                    vtype = VertexType.W_OUTPUT
                elif t == 6:
                    vtype = VertexType.Z_BOX
                else:
                    raise ValueError(f"Unknown vertex type: {t}")

                phase_raw = vertex.get('phase', None)
                phase_frac = None
                if phase_raw is not None:
                    if isinstance(phase_raw, (int, float)):
                        phase_frac = Fraction(phase_raw).limit_denominator()  # interpreted as multiple of π
                    elif isinstance(phase_raw, str):
                        try:
                            # If it’s a π-string like "3π/2" use your parser, else treat as decimal multiple of π
                            phase_frac = pi_string_to_fraction(phase_raw)
                        except Exception:
                            phase_frac = Fraction(phase_raw).limit_denominator()
                vid = g.add_vertex(ty=vtype, phase=phase_frac)
                vertex_ids[vertex['id']] = vid

            # Add undirected edges, map type, avoid duplicates
            seen = set()
            for edge in edges:
                u = vertex_ids[edge['source_id']]
                v = vertex_ids[edge['target_id']]
                if u == v:
                    continue
                key = (min(u, v), max(u, v))
                if key in seen:
                    continue
                seen.add(key)
                etype = zx.EdgeType.HADAMARD if edge.get('t', 1) == 2 else zx.EdgeType.SIMPLE
                g.add_edge(key, edgetype=etype)

            # IO sets
            g.set_inputs([vertex_ids[v['id']] for v in inputs])
            g.set_outputs([vertex_ids[v['id']] for v in outputs])

            # Compute positions (spring layout) so JSON contains "pos"
            nxg = nx.Graph()
            for v in g.vertices():
                nxg.add_node(v)
            for u, v, t in g.edges():
                nxg.add_edge(u, v)
            pos = nx.spring_layout(
                nxg,
                seed=42,
                k=100,  # larger k -> more spacing
                )
            for v, (x, y) in pos.items():
                g.set_position(v, float(x), float(y))

            # Write JSON (to_json returns a JSON string)
            with open(json_file_path, 'w', encoding='utf-8') as f:
                json.dump(json.loads(g.to_json()), f, indent = 4)

            logging.info(f"Graph data exported to {json_file_path}")

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
            uri: URI for the Neo4j/Memgraph instance
            user: Database username
            password: Database password
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
            
        with self.driver.session() as session:
            session.run("CREATE INDEX ON :Node(id);")
        
        # Clear existing graph data if requested
        if initialize_empty:
            with self.driver.session() as session:
                def clear_existing_graph(tx):
                    tx.run("""
                        MATCH (v)
                        DETACH DELETE v
                    """, graph_id=graph_id)
                    
                    return True
                
                session.execute_write(clear_existing_graph)
                
                logging.info(f"Cleared existing graph with ID '{graph_id}'")
        
            with self.driver.session() as session:
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
                                vertex_props["phase"] = float(pi_string_to_fraction(vertex["phase"]))
                            else:
                                if vertex.get("t") != 0:
                                    # Default to 0 if phase is not a valid number or string
                                    vertex_props["phase"] = 0
                            
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
                
                        
            with self.driver.session() as session:
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


    def hadamard_cancel(self, graph_id: str) -> int:
        """
        Cancel Hadamard gates using iterative pattern labeling approach.
        """
        
        with self.driver.session() as session:
            total_patterns = 0
            start_time = time.time()
            # Step 1: Iteratively label patterns
            while True:
                def mark_pattern(tx):
                    # Get the marking query from your JSON collection
                    mark_query = str(self.basic_rewrite_rule_queries["Hadamard cancellation labeling query"]["query"]["code"]["value"])
                    result = tx.run(mark_query)
                    record = result.single()
                    return record["pattern_id"] if record and record["pattern_id"] else None
                
                pattern_id = session.execute_write(mark_pattern)
                if not pattern_id:
                    break  # No more patterns found
                total_patterns += 1
                logging.info(f"Marked pattern {pattern_id} for Hadamard cancellation in graph ID '{graph_id}'")
            
            # Step 2: Process all marked patterns
            if total_patterns > 0:
                def cancel_patterns(tx):
                    cancel_query = str(self.basic_rewrite_rule_queries["Hadamard edge cancellation"]["query"]["code"]["value"])
                    result = tx.run(cancel_query, graph_id=graph_id)
                    return result.single()["patterns_processed"]
                
                processed = session.execute_write(cancel_patterns)
                end_time = time.time()
                logging.info(f"Hadamard cancellation completed in {end_time - start_time} seconds for graph ID '{graph_id}'")
                logging.info(f"Hadamard cancellation: {total_patterns} patterns found, {processed} processed")
            
            return total_patterns


    def remove_identities(self, graph_id: str) -> None:
        """
        Remove identity gates from the graph.
        
        Args:
            graph_id: Identifier for the graph to process
        """

        with self.driver.session() as session:

            start_time = time.time()
            def turn_hadamard_edges_into_gates(tx):

                query = str(self.basic_rewrite_rule_queries["Turn Hadamard edges into Hadamard boxes"]["query"]["code"]["value"])
                tx.run(query, graph_id=graph_id)

                logging.info(f"Hadamard edges turned into gates for graph ID '{graph_id}'")
            
            session.execute_write(turn_hadamard_edges_into_gates)

            def removes_ids(tx):

                query = str(self.basic_rewrite_rule_queries["Remove identities"]["query"]["code"]["value"])
                result = tx.run(query, graph_id=graph_id)
                marked, created, deleted = result.single()

                logging.info(f"Identity cancellation completed for graph ID '{graph_id}' with {marked} marked, {created} created, and {deleted} deleted nodes.")
            
            session.execute_write(removes_ids)
            end_time = time.time()
            logging.info(f"Identity removal completed in {end_time - start_time} seconds for graph ID '{graph_id}'")
    

    def spider_fusion(self, graph_id: str) -> int:
        """
        Perform spider fusion on the graph.
        
        Args:
            graph_id: Identifier for the graph to process
        
        Returns:
            Number of spider fusion patterns processed
        """
        
        with self.driver.session() as session:
                start_time = time.time()
            # Step 1: Iteratively label patterns
            #while True:
                total_patterns = 0
                #processed = 0
                while True:
                    
                    def mark_pattern_green(tx):
                        mark_query = str(self.basic_rewrite_rule_queries["Spider labeling query green"]["query"]["code"]["value"])
                        result = tx.run(mark_query)
                        record = result.single()
                        print(record)
                        return record["pattern_id"] if record and record["pattern_id"] else None
                    
                    pattern_id = session.execute_write(mark_pattern_green)
                    #logging.info(f"Marked green pattern {pattern_id} for spider fusion in graph ID '{graph_id}'")
                    if pattern_id:
                        total_patterns += 1

                    def fuse_spiders(tx):
                        cancel_query = str(self.basic_rewrite_rule_queries["Spider fusion rewrite"]["query"]["code"]["value"])
                        result = tx.run(cancel_query, graph_id=graph_id)
                        return result.single()["patterns_processed"]
                    
                    processed_green = session.execute_write(fuse_spiders)
                    #end_time = time.time()
                    #logging.info(f"Spider fusion completed in {end_time - start_time} seconds for graph ID '{graph_id}'")
                    #logging.info(f"Spider fusion: {total_patterns} patterns found, {processed} processed")

                    def mark_pattern_red(tx):
                        mark_query = str(self.basic_rewrite_rule_queries["Spider labeling query red"]["query"]["code"]["value"])
                        result = tx.run(mark_query)
                        record = result.single()
                        print(record)
                        return record["pattern_id"] if record and record["pattern_id"] else None
                    
                    pattern_id = session.execute_write(mark_pattern_red)
                    #logging.info(f"Marked red pattern {pattern_id} for spider fusion in graph ID '{graph_id}'")
                    

                    def fuse_spiders(tx):
                        cancel_query = str(self.basic_rewrite_rule_queries["Spider fusion rewrite"]["query"]["code"]["value"])
                        result = tx.run(cancel_query, graph_id=graph_id)
                        return result.single()["patterns_processed"]
                    
                    processed_red = session.execute_write(fuse_spiders)
                    #logging.info(f"Spider fusion: {total_patterns} patterns found, {processed} processed")
                    
                    if processed_green == 0 and processed_red == 0:
                        break
                
                def apply_Hopf_rule(tx):
                    hopf_query = str(self.basic_rewrite_rule_queries["Hopf rule"]["query"]["code"]["value"])
                    result = tx.run(hopf_query, graph_id=graph_id)
                    return result.single()["pairs_processed"]
                
                #hopf_processed = session.execute_write(apply_Hopf_rule)
                #logging.info(f"Hopf rule applied for graph ID '{graph_id}' with {hopf_processed} node pairs processed")

                def remove_extra_edges(tx):
                    remove_query = str(self.basic_rewrite_rule_queries["Remove extra edges"]["query"]["code"]["value"])
                    result = tx.run(remove_query, graph_id=graph_id)
                    return result.single()["total_edges_removed"]
                
                #edges_removed = session.execute_write(remove_extra_edges)
                #logging.info(f"Removed {edges_removed} bidirectional edges for graph ID '{graph_id}'")
                end_time = time.time()
                logging.info(f"Spider fusion completed in {end_time - start_time} seconds for graph ID '{graph_id}'")
                return total_patterns
        

    def pivot_rule(self, graph_id: str) -> int:
        """
        Apply the pivot rule to the graph.

        Args:
            graph_id: Identifier for the graph to process

        Returns:
            Number of pivot rule patterns processed
        """

        with self.driver.session() as session:
            start_time = time.time()

            while True:
                processed = 0
                
                def apply_pivot_rule_single_interior_spider(tx):
                    pivot_query = str(self.basic_rewrite_rule_queries["Pivot rule - single interior Pauli spider"]["query"]["code"]["value"])
                    result = tx.run(pivot_query, graph_id=graph_id)
                    return result.single()["interior_pauli_removed"]

                processed = session.execute_write(apply_pivot_rule_single_interior_spider)

                def apply_pivot_rule_two_interior_spiders(tx):
                    pivot_query = str(self.basic_rewrite_rule_queries["Pivot rule - two interior Pauli spiders"]["query"]["code"]["value"])
                    result = tx.run(pivot_query, graph_id=graph_id)
                    return result.single()["pivot_operations_performed"]
                
                processed += session.execute_write(apply_pivot_rule_two_interior_spiders)

                if processed == 0:
                    break

            end_time = time.time()
            logging.info(f"Pivot rule applied for graph ID '{graph_id}' with {processed} patterns processed in {end_time - start_time} seconds")
            return processed
        
        
    def local_complementation_rule(self, graph_id: str) -> int:
        """
        Apply the local complementation rule to the graph.

        Args:
            graph_id: Identifier for the graph to process

        Returns:
            Number of local complementation patterns processed
        """

        with self.driver.session() as session:
            start_time = time.time()
            
            while True:
                while True:

                    def apply_local_complementation_labeling(tx):
                        lc_query = str(self.basic_rewrite_rule_queries["Local complement labeling"]["query"]["code"]["value"])
                        result = tx.run(lc_query, graph_id=graph_id)
                        return result.single()["changed"]
                    changed = session.execute_write(apply_local_complementation_labeling)
                    if changed == 0:
                        break  # No more patterns found
                
                def apply_local_complementation_rewrite(tx):
                    lc_query = str(self.basic_rewrite_rule_queries["Local complement rewrite"]["query"]["code"]["value"])
                    result = tx.run(lc_query, graph_id=graph_id)
                    return result.single()["changed"]
                
                changed = session.execute_write(apply_local_complementation_rewrite)
                if changed == 0:
                    break  # No more patterns found

            end_time = time.time()
            logging.info(f"Local complementation applied for graph ID '{graph_id}' with {changed} patterns processed in {end_time - start_time} seconds")
            return changed
        
    
    def phase_gadget_fusion_rule(self, graph_id: str) -> int:
        """
        Apply the phase gadget fusion rule to the graph.

        Args:
            graph_id: Identifier for the graph to process

        Returns:
            Number of phase gadget fusion patterns processed
        """

        with self.driver.session() as session:
            start_time = time.time()
            
            while True:
                def apply_phase_gadget_fusion_labeling(tx):
                    pgf_query = str(self.basic_rewrite_rule_queries["Gadget fusion red green"]["query"]["code"]["value"])
                    result = tx.run(pgf_query, graph_id=graph_id)
                    return result.single()["fusions_performed"]
                changed = session.execute_write(apply_phase_gadget_fusion_labeling)
                if changed == 0:
                    break  # No more patterns found
            
            while True:
                def apply_phase_gadget_fusion_rewrite(tx):
                    pgf_query = str(self.basic_rewrite_rule_queries["Gadget fusion Hadamard"]["query"]["code"]["value"])
                    result = tx.run(pgf_query, graph_id=graph_id)
                    return result.single()["fusions_performed"]
                
                changed = session.execute_write(apply_phase_gadget_fusion_rewrite)
                if changed == 0:
                    break  # No more patterns found

            end_time = time.time()
            logging.info(f"Phase gadget fusion applied for graph ID '{graph_id}' with {changed} patterns processed in {end_time - start_time} seconds")
            return changed