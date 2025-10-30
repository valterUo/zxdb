# store results to experiments folder
import itertools
import os
import json
from datetime import datetime
import time
import networkx as nx
import numpy as np
from pyzx.circuit.graphparser import circuit_to_graph
from qiskit.quantum_info import Operator
import pyzx as zx
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
from qiskit.quantum_info import Operator

from zxdb.pyzx_utils import edge_matcher, graph_to_quimb_tn, node_matcher, pyzx_to_networkx_manual, pyzx_to_networkx_tensor_graph


def zx_graph_to_db(zxdb, circuit, graph_id="example_graph", json_file="example.json", batch_size=10**6, hadamard_edges = False):
    """
    Converts a circuit to a ZX graph, saves it as JSON, and imports it into the database.
    """
    if type(circuit) == zx.Circuit:
        zx_graph = circuit_to_graph(circuit, compress_rows=False)
    else:
        zx_graph = circuit
    print("Circuit converted to ZX graph.")

    with open(json_file, "w") as f:
        json.dump(json.loads(zx_graph.to_json()), f, indent=4)
    print(f"ZX graph saved to {json_file}.")

    zxdb.import_zx_graph_json_to_graphdb(
        json_file_path=json_file,
        graph_id=graph_id,
        save_metadata=True,
        initialize_empty=True,
        batch_size=batch_size,
        hadamard_edges=hadamard_edges
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
    ty = graph.types()
    degrees = {}
    for v in graph.vertices():
        d = graph.vertex_degree(v)
        if d in degrees: degrees[d] += 1
        else: degrees[d] = 1
    return degrees

def postprocess(zx_graph, zxdb, qubits, test_isomorphism=True):
    """
    Performs post-spider-fusion checks and prints based on qubit count.
    - For <=100 qubits: checks isomorphism between PyZX and DB graphs.
    - For <=500 qubits: prints DB graph stats.
    - For >500 qubits: prints degree distribution.
    """
    db_graph = None
    are_isomorphic = None
    degree_distribution = None

    if qubits <= 200:
        db_graph = zxdb.export_graphdb_to_zx_graph(
            graph_id="example_graph",
            json_file_path="example.json"
        )

    if qubits <= 100 and test_isomorphism:
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
    print("Degree distribution after simplification from DB:")
    if db_graph:
        print(db_graph.stats())
    else:
        for degree, count in degree_distribution.items():
            print(f"Degree {degree}: {count} vertices")
    return degree_distribution, db_graph, are_isomorphic

def tensor_to_unitary_operator(tensor):
    """
    Converts an n-qubit tensor to a Qiskit Operator with proper normalization.
    
    Parameters:
        tensor : np.ndarray
            Tensor of shape (2, 2, ..., 2) representing an n-qubit operator.
    
    Returns:
        Operator : Qiskit Operator instance, properly normalized.
    """
    dim = int(np.prod(tensor.shape[:tensor.ndim//2]))
    matrix = tensor.reshape(dim, dim)
    col_norms = np.linalg.norm(matrix, axis=0)
    matrix_unitary = matrix / col_norms[0]
    return Operator(matrix_unitary)


def check_permutation_equivalence(op_a, op_b):
    """
    Checks if two Operators are equivalent up to a
    permutation of their subsystems.
    """
    if op_a.input_dims() != op_b.input_dims() or \
       op_a.output_dims() != op_b.output_dims():
        return False

    num_qubits = op_a.num_qubits
    qubit_indices = list(range(num_qubits))
    all_perms = list(itertools.permutations(qubit_indices))
    
    print(f"Checking {len(all_perms)} permutations for {num_qubits} qubits...")

    for perm in all_perms:
        op_b_permuted = op_b.apply_permutation(perm)
        if op_a.equiv(op_b_permuted):
            print(f"Found equivalence with permutation: {perm}")
            return True
            
    return False


def get_invariant_signature(qc):
    """Compute permutation-invariant signature"""
    U = Operator(qc).data
    
    # Compute eigenvalues (sorted for comparison)
    eigenvalues = np.linalg.eigvals(U)
    eigenvalues = np.sort(np.abs(eigenvalues))
    
    # Compute trace powers
    traces = [np.trace(np.linalg.matrix_power(U, k)) for k in [1, 2, 3]]
    
    return eigenvalues, traces

def equiv_by_invariants(qc1, qc2, tol=1e-10):
    """Check equivalence using invariants (necessary but not sufficient)"""
    sig1 = get_invariant_signature(qc1)
    sig2 = get_invariant_signature(qc2)
    
    return (np.allclose(sig1[0], sig2[0], atol=tol) and
            np.allclose(sig1[1], sig2[1], atol=tol))


def benchmark_rule(rule_functions, 
                   rule_names, 
                   zx_graph, 
                   zxdb, 
                   qubits,
                   rule,
                   test_isomorphism=True,
                   test_degree_distributions=True,
                   test_tensor_equivalence=True,
                   visualize=False):
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

    original_zx_graph = zx_graph.copy()

    print("Initial stats:", zx_graph.stats())

    if visualize:
        fig = zx.draw(zx_graph)
        rule_name = [name for name in rule_names if "db" not in name][0]
        fig.savefig(f'graph_before_{rule_name}.png')
        plt.close(fig)


    # Run each rule and collect timing/statistics
    for rule_name, rule_fn in zip(rule_names, rule_functions):
        print(f"Running {rule_name}...")
        start_time = time.time()
        if "db" in rule_name:
            result = rule_fn(graph_id="example_graph")
        else:
            result = rule_fn(zx_graph)
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"{rule_name} took:", elapsed, "seconds")
        experiment_data[f'{rule_name}_time'] = elapsed
        experiment_data[f'{rule_name}_result'] = result

    print("Stats from PyZX after simplification: ", zx_graph.stats())

    # Postprocess and compare
    degree_distribution, db_graph, are_isomorphic = postprocess(zx_graph, zxdb, qubits, test_isomorphism)
    experiment_data['final_stats'] = zx_graph.stats()

    if visualize:
        with open("circuits\\graph_after_rule.json", "w") as f:
            json.dump(json.loads(zx_graph.to_json()), f, indent=4)
        fig = zx.draw(zx_graph)
        rule_name = [name for name in rule_names if "db" not in name][0]
        fig.savefig(f'graph_after_{rule_name}.png')
        plt.close(fig)

    if db_graph is not None and visualize:
        
        with open("circuits\\graph_db_after_rule.json", "w") as f:
            json.dump(json.loads(db_graph.to_json()), f, indent=4)

        fig_db = zx.draw(db_graph)
        rule_name = [name for name in rule_names if "db" in name]
        if rule_name:
            rule_name = rule_name[0]
        fig_db.savefig(f'graph_db_after_{rule_name}.png')
        plt.close(fig_db)
    
    if are_isomorphic is not None and test_isomorphism:
        print("Isomorphic:", are_isomorphic)
        #assert are_isomorphic, "Graphs are not structurally equivalent (isomorphic) after rule application."

    degree_dist = get_degree_dist(zx_graph)

    if db_graph is not None and test_degree_distributions:
        assert db_graph.stats() == zx_graph.stats(), "Rule did not reduce the number of vertices as expected."
    elif test_degree_distributions and degree_distribution is not None:
        assert degree_distribution == degree_dist, "Degree distributions do not match after rule application."

    print(zx_graph.stats())


    # For those graphs that in some sense represent circuits, i.e., have the same number of inputs and outputs
    # the tensors should be equivalent up to a permutation of qubits. 
    # The order of the qubits gets mixed in this process.
    if not are_isomorphic and test_tensor_equivalence and len(zx_graph.inputs()) == len(zx_graph.outputs()) and qubits <= 10 and (len(zx_graph.outputs()) + len(zx_graph.inputs())) % 2 == 0:

        if False:
            for g in [zx_graph, db_graph]:
                print("Inputs:", g.inputs())
                print("Outputs:", g.outputs())
                print("Boundary degrees:", [g.vertex_degree(v) for v in g.inputs() + g.outputs()])
                print("Num vertices:", g.num_vertices())
                print("Num edges:", g.num_edges())
        
        # This will not run even for small graphs due to performance issues
        #zx_tensor = zx_graph.to_tensor()
        #db_tensor = db_graph.to_tensor()
        #original_zx_tensor = original_zx_graph.to_tensor()
        #print(zx_tensor.shape)
        #print(db_tensor.shape)
        #print(original_zx_tensor.shape)
        #print(zx.compare_tensors(zx_tensor, original_zx_tensor))
        #print(zx.compare_tensors(db_tensor, original_zx_tensor))
        #db_tensor = db_graph.to_tensor()

        # Quimb performs better, but this does not preserve the scalar
        nx_tn_db = pyzx_to_networkx_tensor_graph(db_graph)
        nx_tn_zx = pyzx_to_networkx_tensor_graph(zx_graph)
        nx_tn_zx_original = pyzx_to_networkx_tensor_graph(original_zx_graph)
        
        quimb_tn_zx = graph_to_quimb_tn(nx_tn_zx)
        quimb_tn_zx.draw(return_fig=True).savefig('tensor_network_pyzx.png')
        quimb_tn_db = graph_to_quimb_tn(nx_tn_db)
        quimb_tn_zx_original = graph_to_quimb_tn(nx_tn_zx_original)
        quimb_tn_zx_original.draw(return_fig=True).savefig('tensor_network_pyzx_original.png')
        
        zx_tensor = quimb_tn_zx.contract().data
        db_tensor = quimb_tn_db.contract().data
        original_zx_tensor = quimb_tn_zx_original.contract().data

        qc1 = tensor_to_unitary_operator(zx_tensor)
        qc2 = tensor_to_unitary_operator(db_tensor)
        qc_original = tensor_to_unitary_operator(original_zx_tensor)

        print("Equivalent by invariants between PyZX and original:", equiv_by_invariants(qc1, qc_original))

        print("Equivalent by invariants between DB and PyZX:", equiv_by_invariants(qc2, qc1))

        #assert equiv_by_invariants(qc1, qc_original), "The operators from the simplified and original PyZX graphs are not equivalent by invariants."
        assert equiv_by_invariants(qc2, qc_original), "The operators from the DB and original PyZX graphs are not equivalent by invariants."
        #assert equiv_by_invariants(qc1, qc2), "The operators from PyZX and DB graphs are not equivalent by invariants."
        
        assert check_permutation_equivalence(qc2, qc_original), "The operators from PyZX and DB graphs are not equivalent."

    store_experiment_results(f'{rule}\\{rule_names[0]}_{qubits}_qubits', experiment_data)