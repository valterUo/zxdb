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
    
    def __init__(self, uri = "bolt://localhost:7687", user = "", password="",
                 storage_mode = None):
        self.uri = uri
        self.user = user
        self.password = password
        # Storage mode. IN_MEMORY_ANALYTICAL is REQUIRED for correctness here:
        # several rewrite queries rely on its IMMEDIATE write visibility within a
        # single query (e.g. identity removal re-checks preconditions per row, so
        # "an earlier removal rewires a later candidate"). IN_MEMORY_TRANSACTIONAL
        # gives each row the start-of-query snapshot instead, so those queries
        # apply mutually-invalidating rewrites in one pass and produce WRONG
        # results (verified: db != original circuit tensor on small cases) — so it
        # is NOT a safe drop-in. The downside of analytical is the absence of
        # snapshot isolation: under heavy/dense mutation a query can dereference a
        # just-deleted node, which surfaces as a "deleted node" error or a
        # non-deterministic SIGSEGV (exit 139). Mitigations: wipe between runs and
        # the auto-recovery in the long-running study scripts. Overridable via
        # storage_mode= / ZXDB_STORAGE_MODE for experimentation only.
        self.storage_mode = (storage_mode
                             or os.environ.get("ZXDB_STORAGE_MODE")
                             or "IN_MEMORY_ANALYTICAL")
        self.basic_rewrite_rule_queries = {}
        self._driver = None
        # When set (by full_reduce / interior_clifford_simp), all rule methods
        # run their fixpoint loop on this shared transaction instead of opening
        # their own. This collapses the dozens of managed transactions a full
        # reduction would otherwise begin/commit into a single one.
        self._active_tx = None

        with open("zxdb/main_queries.json", "r") as f:
            query_collection = json.load(f)
        
        for e in query_collection["items"]:
            self.basic_rewrite_rule_queries[e["title"]] = e

        # Set the storage mode (see __init__ note). Switching mode requires an
        # empty database, which is the case on a fresh connection.
        with self.driver.session() as analyze_session:
            analyze_session.run(f"STORAGE MODE {self.storage_mode};")
            # Indexes for the scans every rewrite rule starts with. The
            # label-property :Node(t) index serves the (n:Node {t: ...}) node
            # scans; the edge-type :Wire(t) index serves the global
            # edge-by-type scans (e.g. the spider-fusion fusable guard). These
            # accelerate finding candidate nodes/edges by value; they do NOT
            # (and cannot) speed up per-node adjacency checks such as the
            # pivot/pivot_gadget interiorness predicates, which must walk a
            # specific spider's neighbours. Idempotent; ignore "already exists".
            for stmt in ("CREATE INDEX ON :Node(t)",
                         "CREATE INDEX ON :Node(graph_id)",
                         "CREATE EDGE INDEX ON :Wire(t)"):
                try:
                    analyze_session.run(stmt).consume()
                except Exception:
                    pass

    def _query(self, title: str) -> str:
        """Cypher text of a rule by its title in main_queries.json."""
        return str(self.basic_rewrite_rule_queries[title]["query"]["code"]["value"])

    def _execute(self, loop):
        """Run a rule's fixpoint `loop(tx)`.

        The loop is handed a SESSION, so every `tx.run(...)` inside it is an
        AUTOCOMMIT query — its own short transaction that is fully committed
        before the next one runs. This is deliberate (not `session.execute_write`,
        which would run the whole fixpoint as ONE long managed transaction): in
        IN_MEMORY_ANALYTICAL mode there is no snapshot isolation, so when many
        mutating statements (including DETACH DELETE) share one transaction a
        later statement can dereference a node an earlier statement already
        deleted — surfacing as a "deleted node" error or a non-deterministic
        SIGSEGV. Autocommit keeps each statement's effects finalized and
        independent, which removes that hazard, and it is behaviour-identical for
        the fixpoint logic (each query still reads the current graph and the loop
        still iterates to a fixpoint). It also avoids managed-transaction retries
        re-running a partially-applied loop over the (un-rolled-back) graph.

        If a shared transaction is active, run on it directly (the caller owns
        the transaction); this path is currently unused by full_reduce.
        """
        if self._active_tx is not None:
            return loop(self._active_tx)
        with self.driver.session() as session:
            return loop(session)

    @staticmethod
    def _run_count(tx, query: str, key: str, graph_id: str) -> int:
        record = tx.run(query, graph_id=graph_id).single()
        return record[key] if record else 0
    
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

    def node_count(self) -> int:
        """Total number of nodes in the database (across all graphs)."""
        with self.driver.session() as session:
            return session.run("MATCH (n) RETURN count(n) AS c").single()["c"]

    def wipe_database(self, batch_size: int = 50000) -> int:
        """
        Delete every node in the database, in batches, and free memory.

        Use this to recover from leaked nodes. Under Memgraph's
        IN_MEMORY_ANALYTICAL storage mode there is no transaction rollback, so
        a server crash (or container stop) in the middle of a rewrite leaves
        the partially-created intermediate nodes behind. Across many such
        interruptions these accumulate, every `MATCH (:Node)` scan slows down,
        and the bloated snapshot makes the next restart heavier — a cycle that
        looks like "the queries got slow". A periodic wipe between unrelated
        workloads keeps scans proportional to the working graph.

        Returns the number of nodes deleted.
        """
        deleted = 0
        with self.driver.session() as session:
            while True:
                n = session.run(
                    "MATCH (n) WITH n LIMIT $b DETACH DELETE n "
                    "RETURN count(n) AS c", b=batch_size).single()["c"]
                deleted += n
                if n == 0:
                    break
            try:
                session.run("FREE MEMORY").consume()
            except Exception:
                pass
        return deleted

    def export_graphdb_to_zx_graph(self,
        graph_id: str,
        json_file_path: str
        ) -> zx.Graph:
        """
        Export a graph from Neo4j or Memgraph database to a PyZX graph and write JSON.
        Positions are computed (spring layout) and stored so they appear as 'pos' in JSON.
        """
        g = zx.Graph()

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
                    RETURN id(v) AS id, v.io_index AS io_index
                """
                output_vertices_query = """
                    MATCH (v:Node:Output {graph_id: $graph_id})
                    RETURN id(v) AS id, v.io_index AS io_index
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
                            # If it's a π-string like "3π/2" use your parser, else treat as decimal multiple of π
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

            # IO sets — restore the original ordering from the stored io_index
            # so the exported graph's input/output tuples line up exactly with
            # the source graph's (no boundary permutation needed downstream).
            # Fall back to DB id order for any boundary without an index.
            def _ordered(records):
                pairs = []
                for rec in records:
                    vid = vertex_ids.get(rec['id'])
                    if vid is None:
                        continue
                    idx = rec.get('io_index')
                    pairs.append((idx if idx is not None else rec['id'], vid))
                pairs.sort(key=lambda p: p[0])
                return tuple(v for _, v in pairs)

            g.set_inputs(_ordered(inputs))
            g.set_outputs(_ordered(outputs))

            # Compute positions (spring layout) so JSON contains "pos"
            nxg = nx.Graph()
            for v in g.vertices():
                nxg.add_node(v)
            for u, v in g.edges():
                nxg.add_edge(u, v)
            pos = nx.spring_layout(
                nxg,
                seed=42,
                k=100,  # larger k -> more spacing
                )
            for v, (x, y) in pos.items():
                g.set_position(v, float(x), float(y))

            #g.normalize()

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
        batch_size: int = 5000,
        hadamard_edges: bool = False
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
            session.run("CREATE INDEX ON :Node(t);")
            session.run("CREATE EDGE INDEX ON :Wire(id);")
            session.run("CREATE EDGE INDEX ON :Wire(t);")
        
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
                    
                    # Mark input vertices with the Input label AND record their
                    # position in the inputs tuple as io_index, so the export can
                    # restore the exact input ordering (boundary nodes survive
                    # every rewrite, so the index stays valid). Preserving the
                    # order means the exported graph's tensor legs line up with
                    # the original's — no boundary-permutation search needed.
                    if "inputs" in graph_data and graph_data["inputs"]:
                        inputs = graph_data["inputs"]
                        ordered = [{"id": vid, "io_index": idx}
                                   for idx, vid in enumerate(inputs)]
                        tx.run("""
                            UNWIND $ordered AS o
                            MATCH (v:Node {graph_id: $graph_id, id: o.id})
                            SET v:Input, v.io_index = o.io_index
                        """, graph_id=graph_id, ordered=ordered)

                    # Same for output vertices.
                    if "outputs" in graph_data and graph_data["outputs"]:
                        outputs = graph_data["outputs"]
                        ordered = [{"id": vid, "io_index": idx}
                                   for idx, vid in enumerate(outputs)]
                        tx.run("""
                            UNWIND $ordered AS o
                            MATCH (v:Node {graph_id: $graph_id, id: o.id})
                            SET v:Output, v.io_index = o.io_index
                        """, graph_id=graph_id, ordered=ordered)
                            
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
        
        if hadamard_edges:
            self.turn_hadamard_gates_into_edges(graph_id=graph_id)
            logging.info(f"Hadamard edges turned into gates for graph ID '{graph_id}'")


    def hadamard_cancel_fn(self, graph_id: str, session) -> int:
        total_patterns = 0
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
        
        # Step 2: Process all marked patterns
        if total_patterns > 0:
            def cancel_patterns(tx):
                cancel_query = str(self.basic_rewrite_rule_queries["Hadamard edge cancellation"]["query"]["code"]["value"])
                result = tx.run(cancel_query, graph_id=graph_id)
                return result.single()["patterns_processed"]
            processed = session.execute_write(cancel_patterns)
        return total_patterns


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


    def remove_identities(self, graph_id: str) -> int:
        """
        Remove identity spiders (phase 0 mod 2, degree 2) from the graph.

        Each removal connects the identity's two neighbors with an edge of the
        combined type (s*s = s, s*h = h, h*h = s). Parallel edges created by a
        removal are normalized the way pyzx's add_edge_table does it: pairs of
        Hadamard edges between same-colored spiders cancel, and a parallel
        simple + Hadamard pair reduces to the simple edge with pi added to one
        endpoint.

        Args:
            graph_id: Identifier for the graph to process

        Returns:
            Number of identity spiders removed
        """
        remove_query = self._query("Remove identities with refactor")

        def loop(tx):
            total = 0
            while True:
                removed = self._run_count(tx, remove_query,
                                          "identities_removed", graph_id)
                self._normalize_parallel_edges_tx(tx, graph_id)
                total += removed
                if removed == 0:
                    return total

        return self._execute(loop)

    def _normalize_parallel_edges_tx(self, tx, graph_id: str) -> int:
        """
        Normalize parallel edges between same-colored spiders the way pyzx's
        add_edge_table does during every rewrite: pairs of Hadamard edges
        cancel, a simple + Hadamard pair becomes the simple edge with pi added
        to one endpoint, and parallel simple edges collapse to one.

        A single cheap guard query short-circuits the common case (no parallel
        edges), which is most calls in the full_reduce pipeline: that turns a
        fixed 6-round-trip cost into 1.
        """
        guard_query = self._query("Parallel edge guard")
        if self._run_count(tx, guard_query, "k", graph_id) == 0:
            return 0

        hadamard_query = self._query("Normalize parallel Hadamard pairs")
        mixed_query = self._query("Normalize parallel mixed pair")
        simple_query = self._query("Normalize parallel simple pairs")
        total = 0
        # Each query is one-shot per node pair, but they interact (a mixed-pair
        # rewrite can leave a Hadamard pair, etc.), so loop until stable.
        while True:
            removed = self._run_count(tx, hadamard_query, "removed", graph_id)
            removed += self._run_count(tx, mixed_query, "removed", graph_id)
            removed += self._run_count(tx, simple_query, "removed", graph_id)
            total += removed
            if removed == 0:
                return total
    

    def spider_fusion(self, graph_id: str) -> int:
        """
        Perform spider fusion on the graph.
        
        Args:
            graph_id: Identifier for the graph to process
        
        Returns:
            Number of spider fusion patterns processed
        """
        fusion_query = self._query("Spider fusion")
        hopf_query = self._query("Hopf")
        # Cheap existence guard: is there any same-colour SIMPLE wire to fuse?
        # Uses the :Wire(t) edge index, so a no-op spider-fusion call costs
        # ~1 ms instead of ~11 ms for the full fusion scan. This is exactly the
        # condition the fusion query acts on, so gating on it is behaviour-
        # identical (the fusion query already returns 0 in that case) — and in
        # a graph-like diagram (post to_gh, all interior wires Hadamard) it is
        # the common case, since pivot/lcomp create only Hadamard wires.
        fusable_guard = (
            "MATCH (a:Node)-[r:Wire {t: 1}]-(b:Node) "
            "WHERE a.t = b.t AND a.t IN [1, 2] "
            "RETURN count(r) AS k")

        def loop(tx):
            # Parallel edges between same-colored spiders must be normalized
            # BEFORE fusing: a fusable pair connected by both a simple and a
            # Hadamard edge fuses to a spider with an extra pi (Hadamard
            # self-loop); the fusion query would silently drop the second edge
            # otherwise. Only the input can carry such pairs up front; each
            # later normalization runs only after a merge actually creates new
            # parallel edges, so no-op fusion calls cost just one fuse query.
            self._normalize_parallel_edges_tx(tx, graph_id)
            total_patterns = 0
            while True:
                if self._run_count(tx, fusable_guard, "k", graph_id) == 0:
                    return total_patterns
                merged = self._run_count(tx, fusion_query, "merged", graph_id)
                if merged == 0:
                    return total_patterns
                total_patterns += merged
                # A merge can create parallel Z-X simple wires (Hopf) and
                # same-colour parallel wires (normalize); clean both up before
                # the next fuse.
                tx.run(hopf_query, graph_id=graph_id).consume()
                self._normalize_parallel_edges_tx(tx, graph_id)

        return self._execute(loop)
        

    def pivot_rule(self, graph_id: str) -> int:
        """
        Apply the pivot rule to the graph.

        Args:
            graph_id: Identifier for the graph to process

        Returns:
            Number of pivot rule patterns processed
        """

        two_query = self._query("Pivot rule - two interior Pauli spiders")
        single_query = self._query("Pivot rule - single interior Pauli spider")

        def loop(tx):
            # Run interior-interior pivots to a fixpoint first, then check for a
            # boundary pivot. The boundary-pivot query is a relatively expensive
            # full scan that rarely matches, so running it once per
            # interior-fixpoint instead of after every single interior pivot
            # avoids dozens of no-op scans. Pivots are confluent, so the order
            # does not change the final graph.
            total = 0
            while True:
                while True:
                    n = self._run_count(
                        tx, two_query, "pivot_operations_performed", graph_id)
                    total += n
                    if n == 0:
                        break
                b = self._run_count(
                    tx, single_query, "interior_pauli_removed", graph_id)
                total += b
                if b == 0:
                    return total

        return self._execute(loop)
        
        
    def local_complementation_rule(self, graph_id: str) -> int:
        """
        Apply the local complementation rule to the graph.

        Args:
            graph_id: Identifier for the graph to process

        Returns:
            Number of local complementation patterns processed
        """

        lc_query = self._query("Local complement")

        def loop(tx):
            total = 0
            while True:
                changed = self._run_count(tx, lc_query,
                                          "patterns_processed", graph_id)
                total += changed
                if changed == 0:
                    return total

        return self._execute(loop)
        
    
    def phase_gadget_fusion_rule(self, graph_id: str) -> int:
        """
        Apply the phase gadget fusion rule to the graph.

        Args:
            graph_id: Identifier for the graph to process

        Returns:
            Number of phase gadget fusion patterns processed
        """

        norm_query = self._query("Gadget axel normalization")
        fusion_query = self._query("Gadget fusion")

        def loop(tx):
            total = 0
            while True:
                self._run_count(tx, norm_query, "normalized", graph_id)
                changed = self._run_count(tx, fusion_query,
                                          "fusions_performed", graph_id)
                total += changed
                if changed == 0:
                    return total

        return self._execute(loop)
        

    def pivot_gadget_rule(self, graph_id: str) -> int:
        """
        Apply the pivot gadget rule to the graph.

        Args:
            graph_id: Identifier for the graph to process

        Returns:
            Number of patterns processed
        """

        pg_query = self._query("Pivot gadget")

        def loop(tx):
            total = 0
            while True:
                changed = self._run_count(
                    tx, pg_query, "pivot_operations_performed", graph_id)
                total += changed
                if changed == 0:
                    return total

        return self._execute(loop)
        

    def pivot_boundary_rule(self, graph_id: str) -> int:
        """
        Apply the pivot boundary rule to the graph.

        Args:
            graph_id: Identifier for the graph to process

        Returns:
            Number of patterns processed
        """

        pb_query = self._query("Pivot boundary")

        def loop(tx):
            total = 0
            while True:
                changed = self._run_count(
                    tx, pb_query, "pivot_operations_performed", graph_id)
                total += changed
                if changed == 0:
                    return total

        return self._execute(loop)


    def bialgebra_simp(self, graph_id: str) -> int:
        """
        Apply the bialgebra rule to the graph.

        Args:
            graph_id: Identifier for the graph to process

        Returns:
            Number of patterns processed
        """

        rule_query = self._query("Bialgebra rule")
        connect_query = self._query("Bialgebra connect")

        def loop(tx):
            total = 0
            while True:
                applied = self._run_count(tx, rule_query,
                                          "bialg_applied", graph_id)
                if applied:
                    tx.run(connect_query, graph_id=graph_id).consume()
                total += applied
                if applied == 0:
                    return total

        return self._execute(loop)
        

    def supplementarity_simp(self, graph_id: str) -> int:
        """
        Apply the supplementarity rule to the graph.

        Finds pairs of non-Clifford Z-spiders whose phases sum to pi (mod 2pi)
        and that share exactly the same neighborhood. Both spiders are removed
        and pi is added to each shared neighbor's phase.

        Args:
            graph_id: Identifier for the graph to process

        Returns:
            Total number of supplementarity applications performed
        """
        type1_query = self._query("Supplementarity type 1")
        type2_query = self._query("Supplementarity type 2")
        # The pair-matching queries are quadratic in the number of
        # non-Clifford spiders; skip them when fewer than two exist.
        guard_query = (
            "MATCH (v:Node {t: 1}) "
            "WHERE v.phase IS NOT NULL AND v.phase * 2 <> round(v.phase * 2) "
            "RETURN count(v) AS k")

        def loop(tx):
            total = 0
            while True:
                if self._run_count(tx, guard_query, "k", graph_id) < 2:
                    return total
                applied = self._run_count(
                    tx, type1_query, "supplementarity_applied", graph_id)
                applied += self._run_count(
                    tx, type2_query, "supplementarity_applied", graph_id)
                total += applied
                if applied == 0:
                    return total

        return self._execute(loop)

    def copy_simp(self, graph_id: str) -> int:
        """
        Apply the copy rule (pyzx >= 0.10 semantics).

        An arity-1 spider with phase 0 or pi is copied through its neighbor:
        every other wire of the neighbor receives a new spider carrying the
        leaf's phase (toggled color across a Hadamard wire, same color across
        a simple wire); the leaf and the neighbor are deleted.

        Args:
            graph_id: Identifier for the graph to process

        Returns:
            Total number of copy rule applications performed
        """
        copy_query = self._query("Copy rule")

        def loop(tx):
            total = 0
            while True:
                applied = self._run_count(tx, copy_query,
                                          "copy_applied", graph_id)
                total += applied
                if applied == 0:
                    return total

        return self._execute(loop)

    def to_gh(self, graph_id: str) -> int:
        """
        Color change (pyzx to_gh): turn every X-spider into a Z-spider by
        toggling the types of its incident wires. A wire between two X-spiders
        toggles twice and therefore keeps its type.

        Returns:
            Number of recolored spiders
        """
        # Autocommit (see _execute): each statement is its own committed
        # transaction, avoiding the analytical-mode deleted-node hazard.
        with self.driver.session() as session:
            # Cheap guard: to_gh only acts on X-spiders. After the first color
            # change the diagram is all Z, so every later call is a no-op — gate
            # the toggle+recolor on a single existence check (behaviour-identical;
            # both queries return 0 when no X-spider exists).
            if session.run("MATCH (n:Node {t: 2}) WITH n LIMIT 1 "
                           "RETURN count(n) AS has").single()["has"] == 0:
                return 0
            toggle_query = str(self.basic_rewrite_rule_queries["Color change edge toggle"]["query"]["code"]["value"])
            session.run(toggle_query, graph_id=graph_id).consume()
            recolor_query = str(self.basic_rewrite_rule_queries["Color change spiders"]["query"]["code"]["value"])
            record = session.run(recolor_query, graph_id=graph_id).single()
            return record["recolored"] if record else 0

    def interior_clifford_simp(self, graph_id: str) -> int:
        """
        pyzx interior_clifford_simp: spider fusion, color change to a
        graph-like diagram, then rounds of identity removal, spider fusion,
        pivot and local complementation until none of them applies.

        Returns:
            Number of full rounds in which at least one rule fired
        """
        self.spider_fusion(graph_id)
        self.to_gh(graph_id)
        rounds = 0
        while True:
            applied = self.remove_identities(graph_id)
            applied += self.spider_fusion(graph_id)
            applied += self.pivot_rule(graph_id)
            applied += self.local_complementation_rule(graph_id)
            if applied == 0:
                break
            rounds += 1
        return rounds

    def clifford_simp(self, graph_id: str) -> int:
        """
        pyzx clifford_simp: rounds of interior_clifford_simp and
        pivot_boundary_simp until neither applies.
        """
        total = 0
        while True:
            total += self.interior_clifford_simp(graph_id)
            if self.pivot_boundary_rule(graph_id) == 0:
                break
        return total

    def remove_isolated_vertices(self, graph_id: str) -> int:
        """Remove degree-0 nodes (pyzx remove_isolated_vertices)."""
        remove_query = self._query("Remove isolated vertices")

        def loop(tx):
            return self._run_count(tx, remove_query, "removed", graph_id)

        return self._execute(loop)

    def _full_reduce_body(self, graph_id: str, max_rounds: int) -> None:
        self.interior_clifford_simp(graph_id)
        self.pivot_gadget_rule(graph_id)
        for _ in range(max_rounds):
            self.clifford_simp(graph_id)
            i = self.phase_gadget_fusion_rule(graph_id)
            self.interior_clifford_simp(graph_id)
            k = self.copy_simp(graph_id)
            l = self.supplementarity_simp(graph_id)
            j = self.pivot_gadget_rule(graph_id)
            if i + j + k + l == 0:
                self.remove_isolated_vertices(graph_id)
                return
        raise RuntimeError(
            f"full_reduce did not converge within {max_rounds} rounds "
            f"for graph ID '{graph_id}'")

    def full_reduce(self, graph_id: str, max_rounds: int = 1000) -> None:
        """
        pyzx full_reduce (pyzx >= 0.10): interior_clifford_simp and
        pivot_gadget_simp once, then rounds of clifford_simp, gadget_simp,
        interior_clifford_simp, copy_simp, supplementarity_simp and
        pivot_gadget_simp until none of the gadget/copy/supplementarity rules
        fires; finally removes isolated vertices.

        Each rule applies in its own write transaction. A single shared
        transaction across the whole reduction (the _active_tx path) was
        measured to give no speedup — full_reduce is bound by per-query
        execution (graph scans), not by transaction begin/commit overhead — so
        per-rule transactions are used to avoid any long-transaction risk.

        Args:
            graph_id: Identifier for the graph to process
            max_rounds: Safety cap on the outer loop (pyzx has none; this
                turns a potential non-termination bug into a visible error)
        """
        self._full_reduce_body(graph_id, max_rounds)

    # ------------------------------------------------------------------
    # Batched-pivot variant (isolated; research). full_reduce_batched_pivot
    # mirrors full_reduce exactly but swaps the interior pivot for the batched
    # query "Pivot rule - batched interior Pauli spiders", which applies a
    # maximal set of disjoint pivots per round-trip instead of one (LIMIT 1).
    # Everything else (boundary pivot, gadget/copy/supplementarity, the whole
    # composition) is unchanged, and the original full_reduce path is untouched.
    # ------------------------------------------------------------------

    def pivot_rule_batched(self, graph_id: str) -> int:
        """Interior pivots applied in disjoint batches (see the batched query),
        then the boundary pivot — same fixpoint as pivot_rule, fewer round-trips."""
        two_query = self._query("Pivot rule - batched interior Pauli spiders")
        single_query = self._query("Pivot rule - single interior Pauli spider")

        def loop(tx):
            total = 0
            while True:
                while True:
                    n = self._run_count(
                        tx, two_query, "pivot_operations_performed", graph_id)
                    total += n
                    if n == 0:
                        break
                b = self._run_count(
                    tx, single_query, "interior_pauli_removed", graph_id)
                total += b
                if b == 0:
                    return total

        return self._execute(loop)

    def interior_clifford_simp_batched(self, graph_id: str) -> int:
        """interior_clifford_simp using the batched interior pivot."""
        self.spider_fusion(graph_id)
        self.to_gh(graph_id)
        rounds = 0
        while True:
            applied = self.remove_identities(graph_id)
            applied += self.spider_fusion(graph_id)
            applied += self.pivot_rule_batched(graph_id)
            applied += self.local_complementation_rule(graph_id)
            if applied == 0:
                break
            rounds += 1
        return rounds

    def clifford_simp_batched(self, graph_id: str) -> int:
        """clifford_simp using the batched interior pivot."""
        total = 0
        while True:
            total += self.interior_clifford_simp_batched(graph_id)
            if self.pivot_boundary_rule(graph_id) == 0:
                break
        return total

    def full_reduce_batched_pivot(self, graph_id: str,
                                  max_rounds: int = 1000) -> None:
        """full_reduce with the batched interior pivot (isolated research variant).

        Identical composition to full_reduce; only the interior pivot rule is
        the batched query, which applies a maximal set of pivots with pairwise
        DISJOINT closed neighbourhoods per round-trip (disjoint pivots commute,
        so the batch equals applying them one-by-one; the global-minimum-rank
        candidate always survives, so progress is guaranteed). CORRECTNESS is
        verified equal to full_reduce on 34 diverse cases
        (evaluation/validate_batched_pivot.py).

        EMPIRICAL FINDING (2026-06-14): this is NOT faster than full_reduce and
        is usually much SLOWER (q8 2x, q16 ~9x), for two independent reasons:
        (1) typical circuits have FEW simultaneously-disjoint interior pivots
        (neighbourhoods overlap heavily), so batches are tiny (~1-2 pivots) and
        round-trips barely drop; (2) the disjoint-set selection cost dominates —
        full_reduce is transport-bound (~1 ms/round-trip) only because each rule
        query's EXECUTION is cheap, and the selection (an O(P^2) list-join, since
        Memgraph has no hash-join; an O(E) temp-property variant is blocked by
        "UNWIND can't follow an update") makes execution expensive, outweighing
        the few round-trips saved. Kept isolated for research; full_reduce is the
        production path.
        """
        self.interior_clifford_simp_batched(graph_id)
        self.pivot_gadget_rule(graph_id)
        for _ in range(max_rounds):
            self.clifford_simp_batched(graph_id)
            i = self.phase_gadget_fusion_rule(graph_id)
            self.interior_clifford_simp_batched(graph_id)
            k = self.copy_simp(graph_id)
            l = self.supplementarity_simp(graph_id)
            j = self.pivot_gadget_rule(graph_id)
            if i + j + k + l == 0:
                self.remove_isolated_vertices(graph_id)
                return
        raise RuntimeError(
            f"full_reduce_batched_pivot did not converge within {max_rounds} "
            f"rounds for graph ID '{graph_id}'")

    # ------------------------------------------------------------------
    # Query-module variant (isolated; research). full_reduce_with_query_modules
    # mirrors full_reduce exactly but runs the pivot_gadget rule via the
    # in-process C-Python query module zxqm (zxdb_qm/zxqm.py, deployed to the
    # Memgraph query-modules dir). That module ports pyzx's match_pivot_gadget +
    # pivot: it runs the WHOLE pivot_gadget fixpoint in one CALL, using pyzx's
    # greedy maximal-disjoint-set matching (mark consumed neighbourhoods) — the
    # imperative O(E) batching that declarative Cypher / MAGE coloring can't do
    # cheaply. Everything else stays Cypher. Requires the module to be loaded
    # (CALL mg.load_all() after deploying zxqm.py); ensure_query_module() does it.
    # ------------------------------------------------------------------

    def ensure_query_module(self) -> bool:
        """Load query modules so zxqm.pivot_gadget_fixpoint is callable.
        Returns True if the proc is available."""
        with self.driver.session() as session:
            try:
                session.run("CALL mg.load_all()").consume()
            except Exception:
                pass
            n = session.run(
                "CALL mg.procedures() YIELD name "
                "WITH name WHERE name = 'zxqmcpp.pivot_gadget_fixpoint' "
                "RETURN count(*) AS c").single()["c"]
            return n > 0

    def pivot_gadget_rule_qm(self, graph_id: str) -> int:
        """pivot_gadget to a fixpoint via the in-process C++ query module zxqmcpp
        (one CALL, no per-application round-trips; pyzx-style greedy maximal
        disjoint matching done natively). Same fixpoint as pivot_gadget_rule. (A
        Python port zxqm.pivot_gadget_fixpoint($graph_id) also exists but is
        slower — the mgp Python per-op overhead exceeds the round-trip savings.)"""
        def loop(tx):
            rec = tx.run(
                "CALL zxqmcpp.pivot_gadget_fixpoint() "
                "YIELD applied RETURN applied").single()
            return rec["applied"] if rec else 0
        return self._execute(loop)

    def full_reduce_with_query_modules(self, graph_id: str,
                                       max_rounds: int = 1000) -> None:
        """full_reduce with pivot_gadget run by the zxqm query module (isolated
        research variant). Identical composition to full_reduce; only the
        pivot_gadget rule is the in-process module."""
        self.interior_clifford_simp(graph_id)
        self.pivot_gadget_rule_qm(graph_id)
        for _ in range(max_rounds):
            self.clifford_simp(graph_id)
            i = self.phase_gadget_fusion_rule(graph_id)
            self.interior_clifford_simp(graph_id)
            k = self.copy_simp(graph_id)
            l = self.supplementarity_simp(graph_id)
            j = self.pivot_gadget_rule_qm(graph_id)
            if i + j + k + l == 0:
                self.remove_isolated_vertices(graph_id)
                return
        raise RuntimeError(
            f"full_reduce_with_query_modules did not converge within "
            f"{max_rounds} rounds for graph ID '{graph_id}'")

    def get_degree_distribution(self, graph_id: str) -> dict:
        """
        Get the degree distribution of the graph.

        Args:
            graph_id: Identifier for the graph to process

        Returns:
            Dictionary mapping degree to count of vertices with that degree
        """

        degree_distribution = {}

        with self.driver.session() as session:
            def fetch_degree_distribution(tx):
                query = str(self.basic_rewrite_rule_queries["Get degree distribution"]["query"]["code"]["value"])
                result = tx.run(query, graph_id=graph_id)
                return {record["degree"]: record["frequency"] for record in result}

            degree_distribution = session.execute_read(fetch_degree_distribution)

        return degree_distribution
    

    def turn_hadamard_gates_into_edges(self, graph_id: str) -> None:
        """
        Turn Hadamard gates into edges in the graph.

        Args:
            graph_id: Identifier for the graph to process
        """

        with self.driver.session() as session:
            def apply_hadamard_to_edge_conversion(tx):
                query = str(self.basic_rewrite_rule_queries["Turn Hadamard gates into edges"]["query"]["code"]["value"])
                tx.run(query, graph_id=graph_id)

            session.execute_write(apply_hadamard_to_edge_conversion)