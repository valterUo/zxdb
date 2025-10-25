import json
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

import numpy as np
from qiskit import QuantumCircuit

from pyzx import Graph
from pyzx.graph.multigraph import Multigraph
from pyzx.graph.jsonparser import json_to_graph_old
from pyzx import string_to_phase
import pyzx as zx
import pyzx as zx
from pyzx.circuit import Circuit
from fractions import Fraction

import networkx as nx

from fractions import Fraction

def pi_string_to_fraction(s: str) -> Fraction:
    s = s.replace(" ", "")  # Remove spaces
    if 'π' not in s:
        return Fraction(s)  # No pi, treat as regular number

    if s == 'π':
        return Fraction(1)

    if s.startswith('π/'):
        return Fraction(1, int(s.split('/')[1]))

    if s.endswith('π'):
        return Fraction(int(s[:-1]))

    if 'π/' in s:
        num = s.split('π/')[0]
        denom = s.split('π/')[1]
        return Fraction(int(num), int(denom))

    raise ValueError(f"Unrecognized format: {s}")


def dict_to_graph(d, backend=None):
    """Converts a Python dict representation a graph produced by `graph_to_dict` into
    a pyzx Graph.
    If backend is given, it will be used as the backend for the graph, 
    otherwise the backend will be read from the dict description."""

    if not 'version' in d:
        # "Version is not specified in dictionary, will try to parse it as an older format")
        return json_to_graph_old(d, backend)
    else:
        if d['version'] != 2:
            raise ValueError("Unsupported version "+str(d['version']))
        
    if backend == None:
        backend = d.get('backend', None)
        if backend is None: raise ValueError("No backend specified in dictionary")
    
    g = Graph(backend)
    g.variable_types = d.get('variable_types',{})
    g._inputs = d.get('inputs', [])
    g._outputs = d.get('outputs', [])
    if g.backend == 'multigraph':
        if TYPE_CHECKING:
            assert isinstance(g, Multigraph)
        b = True if d.get('auto_simplify', True) in ('true', True) else False
        g.set_auto_simplify(b)
    for v_d in d['vertices']:
        pos = v_d['pos']
        v = v_d['id']
        g.add_vertex_indexed(v)
        g.set_type(v,v_d['t'])
        g.set_row(v,pos[0])
        g.set_qubit(v,pos[1])
        if 'phase' in v_d:
            g.set_phase(v,string_to_phase(v_d['phase'],g))
        if 'is_ground' in v_d and v_d['is_ground'] == True:
            g.set_ground(v)
        if 'data' in v_d:
            for k,val in v_d['data'].items():
                g.set_vdata(v,k,val)

    for (s,t,et) in d['edges']:
        g.add_edge((s,t),et)

    return g

def json_to_graph(js, backend=None):

    """Converts the json representation of a pyzx graph (as a string or dict) into
    a `Graph`. If JSON is given as a string, parse it first."""
    if isinstance(js, str):
        d = json.loads(js)
    else:
        d = js
    return dict_to_graph(d, backend)

def extract_circuit_from_zx_graph(G: nx.Graph):
    """
    Simplified version of ZX-diagram circuit extraction.
    Input: G - a NetworkX graph representing a ZX-diagram.
    Output: Qiskit QuantumCircuit.
    """
    # 1. Identify input and output nodes
    input_nodes = [n for n, d in G.nodes(data=True) if d.get("is_input")]
    output_nodes = [n for n, d in G.nodes(data=True) if d.get("is_output")]

    n_qubits = len(output_nodes)
    qc = QuantumCircuit(n_qubits)

    # 2. Build initial "frontier": the neighbor of each output (excluding outputs)
    frontier = []
    wire_map = {}
    for i, out in enumerate(output_nodes):
        neighbors = [n for n in G.neighbors(out) if not G.nodes[n].get("is_output")]
        if neighbors:
            v = neighbors[0]
            frontier.append(v)
            wire_map[v] = i

    # 3. Main extraction loop
    while frontier:
        v = frontier.pop(0)
        vdata = G.nodes[v]
        q = wire_map[v]

        # 3a. Apply Hadamard if any incident edge is hadamard
        for neighbor in list(G.neighbors(v)):
            edata = G.get_edge_data(v, neighbor)
            if edata and edata.get("type") == "hadamard":
                qc.h(q)
                G.remove_edge(v, neighbor)

        # 3b. Apply phase if present
        phase = vdata.get("phase", 0)
        if phase != 0:
            if vdata.get("type") == "Z":
                qc.rz(phase, q)
            elif vdata.get("type") == "X":
                qc.rx(phase, q)
        
        # 3c. Check for 2-qubit connections (CNOT/CZ)
        for neighbor in list(G.neighbors(v)):
            if neighbor in wire_map:
                q2 = wire_map[neighbor]
                edata = G.get_edge_data(v, neighbor)
                if vdata.get("type") == "Z" and G.nodes[neighbor].get("type") == "X":
                    qc.cx(q, q2)
                elif vdata.get("type") == "Z" and G.nodes[neighbor].get("type") == "Z":
                    qc.cz(q, q2)
                elif vdata.get("type") == "X" and G.nodes[neighbor].get("type") == "X":
                    qc.cx(q, q2)  # For simplicity; real ZX may use XCX
                G.remove_edge(v, neighbor)

        # 3d. Remove node, update frontier
        for neighbor in list(G.neighbors(v)):
            if not G.nodes[neighbor].get("is_input") and neighbor not in frontier:
                frontier.append(neighbor)
                wire_map[neighbor] = q
        G.remove_node(v)

    # 4. Final swaps/Hadamards for correct output mapping (optional, not implemented here)
    return qc


def pyzx_to_networkx_manual(g: zx.Graph):
    """Manually converts a pyzx.Graph to a networkx.Graph, preserving properties."""
    nxg = nx.Graph()
    for v in g.vertices():
        nxg.add_node(
            v, 
            type=g.type(v), 
            phase=g.phase(v)
        )
    for e in g.edges():
        source, target = g.edge_s(e), g.edge_t(e)
        nxg.add_edge(
            source, 
            target, 
            type=g.edge_type(e)
        )
    return nxg

def node_matcher(node1_data, node2_data):
    """Returns True if nodes have the same type and phase."""
    if node1_data.get('type') != node2_data.get('type'):
        return False
    phase1 = node1_data.get('phase', 0)
    phase2 = node2_data.get('phase', 0)
    # Use a tolerance for comparing floating point phases if necessary
    #print(f"Comparing phases: {phase1} and {phase2} with types {type(phase1)} and {type(phase2)}")
    if type(phase1) == Fraction and type(phase2) == Fraction:
        return phase1 == phase2
    elif type(phase1) == float and type(phase2) == float:
        return np.isclose(phase1, phase2, atol=1e-9)
    elif type(phase1) == int and type(phase2) == Fraction:
        return np.isclose(float(phase1), float(phase2), atol=1e-9)
    elif type(phase1) == Fraction and type(phase2) == int:
        return np.isclose(float(phase1), float(phase2), atol=1e-9)
    are_close = np.isclose(phase1, phase2, atol=1e-9)
    #if not are_close:
    #    print(f"Phase mismatch: {phase1} vs {phase2}")
    return are_close


def edge_matcher(edge1_data, edge2_data):
    """Returns True if edges have the same type."""
    #print(f"Comparing edge types: {edge1_data.get('type')} and {edge2_data.get('type')}")
    return edge1_data.get('type') == edge2_data.get('type')


def phase_poly_term_to_graph(
    coeff: float | Fraction,
    interactions: list[int],
    num_qubits: int,
    prev_circuit: Circuit | None = None
) -> zx.Graph:
    """
    Creates a ZX-diagram for a single term of a phase polynomial.

    The term corresponds to the unitary exp(i * coeff * Z_i * Z_j * ...),
    where i, j, ... are the qubits in the `interactions` list.

    This is implemented by creating a CNOT ladder to compute the parity,
    applying an Rz rotation, and then un-computing the CNOT ladder.

    Args:
        coeff: The phase coefficient of the term (alpha).
        interactions: A list of qubit indices involved in the Pauli Z product.
        num_qubits: The total number of qubits in the circuit.

    Returns:
        A pyzx.Graph representing the circuit for this term.
    """
    interactions = sorted(list(set(interactions)))

    if prev_circuit is None:
        c = Circuit(num_qubits)
    else:
        c = prev_circuit

    if len(interactions) == 1:
        qubit = interactions[0]
        c.add_gate("ZPhase", qubit, phase=2 * coeff)

    elif len(interactions) > 1:
        target_qubit = interactions[-1]

        for i in range(len(interactions) - 1):
            control = interactions[i]
            target = interactions[i+1]
            c.add_gate("CNOT", control, target)

        c.add_gate("ZPhase", target_qubit, phase=2 * coeff)

        for i in range(len(interactions) - 2, -1, -1):
            control = interactions[i]
            target = interactions[i+1]
            c.add_gate("CNOT", control, target)
    else:
        pass

    #g = c.to_graph()
    
    #if not interactions:
    #    g.scalar.add_phase(coeff)

    return c