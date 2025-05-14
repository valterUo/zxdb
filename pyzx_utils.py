import json
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

from pyzx import Graph
from pyzx.graph.multigraph import Multigraph
from pyzx.graph.jsonparser import json_to_graph_old
from pyzx import string_to_phase


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