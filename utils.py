# store results to experiments folder
import os
import json
from datetime import datetime
import time
import networkx as nx
from pyzx.circuit.graphparser import circuit_to_graph

from zxdb.pyzx_utils import edge_matcher, node_matcher, pyzx_to_networkx_manual

def zx_graph_to_db(zxdb, circuit, graph_id="example_graph", json_file="example.json", batch_size=10**6):
    """
    Converts a circuit to a ZX graph, saves it as JSON, and imports it into the database.
    """
    zx_graph = circuit_to_graph(circuit, compress_rows=False)
    print("Circuit converted to ZX graph.")

    with open(json_file, "w") as f:
        json.dump(json.loads(zx_graph.to_json()), f, indent=4)
    print(f"ZX graph saved to {json_file}.")

    zxdb.import_zx_graph_json_to_graphdb(
        json_file_path=json_file,
        graph_id=graph_id,
        save_metadata=True,
        initialize_empty=True,
        batch_size=batch_size
    )
    print(f"ZX graph imported to database as '{graph_id}'.")
    return zx_graph

def store_experiment_results(experiment_name: str, data: dict):

    if not os.path.exists('experiments'):
        os.makedirs('experiments')

    # Create a timestamped filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{experiment_name}_{timestamp}.json"
    filepath = os.path.join('experiments', filename)

    # Write data to JSON file
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"Experiment results saved to {filepath}")

def get_degree_dist(graph):
    degrees = {}
    for v in graph.vertices():
        d = graph.vertex_degree(v)
        if d in degrees: degrees[d] += 1
        else: degrees[d] = 1
    return degrees

def postprocess(zx_graph, zxdb, qubits):
    """
    Performs post-spider-fusion checks and prints based on qubit count.
    - For <=100 qubits: checks isomorphism between PyZX and DB graphs.
    - For <=500 qubits: prints DB graph stats.
    - For >500 qubits: prints degree distribution.
    """
    db_graph = None
    are_isomorphic = None

    if qubits <= 500:
        db_graph = zxdb.export_graphdb_to_zx_graph(
            graph_id="example_graph",
            json_file_path="example.json"
        )
        print(db_graph.stats())

    if qubits <= 100:
        nxg1 = pyzx_to_networkx_manual(zx_graph)
        nxg2 = pyzx_to_networkx_manual(db_graph)
        are_isomorphic = nx.is_isomorphic(
            nxg1, 
            nxg2, 
            node_match=node_matcher, 
            edge_match=edge_matcher
        )
        print("Isomorphic:", are_isomorphic)

    degree_distribution = zxdb.get_degree_distribution(graph_id="example_graph")
    print("Degree distribution after simplification:")
    for degree, count in degree_distribution.items():
        print(f"Degree {degree}: {count} vertices")
    return degree_distribution, db_graph, are_isomorphic

def benchmark_rule(rule_functions, rule_names, zx_graph, zxdb, qubits):
    """
    Generic benchmarking for ZX rules.
    - rule_functions: list of callables to apply (e.g. [zxdb.spider_fusion, zx.spider_simp])
    - rule_names: list of names for each rule (e.g. ["db_spider_fusion", "pyzx_spider_fusion"])
    - zx_graph: PyZX graph object
    - zxdb: ZXdb instance
    - qubits: number of qubits
    - get_degree_dist_fn: function for degree distribution (e.g. get_degree_dist)
    - store_results_fn: function to store results (e.g. store_experiment_results)
    """
    experiment_data = {}
    experiment_data['qubits'] = qubits
    experiment_data['initial_stats'] = zx_graph.stats()

    # Run each rule and collect timing/statistics
    for rule_name, rule_fn in zip(rule_names, rule_functions):
        print(f"Running {rule_name}...")
        start_time = time.time()
        result = rule_fn(graph_id="example_graph") if "db" in rule_name else rule_fn(zx_graph)
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"{rule_name} took:", elapsed, "seconds")
        experiment_data[f'{rule_name}_time'] = elapsed
        experiment_data[f'{rule_name}_result'] = result

    # Postprocess and compare
    degree_distribution, db_graph, are_isomorphic = postprocess(zx_graph, zxdb, qubits)
    experiment_data['final_stats'] = zx_graph.stats()

    if are_isomorphic is not None:
        print("Isomorphic:", are_isomorphic)
        assert are_isomorphic, "Graphs are not structurally equivalent (isomorphic) after rule application."

    print(zx_graph.stats())
    degree_dist = get_degree_dist(zx_graph)

    if db_graph is not None:
        assert db_graph.stats() == zx_graph.stats(), "Rule did not reduce the number of vertices as expected."
    else:
        assert degree_distribution == degree_dist, "Degree distributions do not match after rule application."

    # Store results with correct naming
    store_experiment_results(f'{rule_names[0]}_{qubits}_qubits', experiment_data)