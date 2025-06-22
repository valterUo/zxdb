import json
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

from qiskit import QuantumCircuit

from pyzx import Graph
from pyzx.graph.multigraph import Multigraph
from pyzx.graph.jsonparser import json_to_graph_old
from pyzx import string_to_phase

import networkx as nx


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