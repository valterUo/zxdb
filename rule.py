import networkx as nx


class Rule:
    
    def __init__(self, name, lhs, rhs):
        self.name = name
        assert isinstance(lhs, nx.DiGraph), "LHS must be a directed acyclic graph (DAG)"
        assert isinstance(rhs, nx.DiGraph), "RHS must be a directed acyclic graph (DAG)"
        self.lhs = lhs
        self.rhs = rhs
        self.node_vars = set()
        self.edge_vars = set()
        self.cypher_lines_match = []
        
    def get_lhs_pattern(self):
        """
        Get the left-hand side pattern of the rule.
        """
        # Assuming lhs is a DAG, traverse it along the edges to get the pattern

        # Find sinks in the original graph (sources in the reversed one)
        sinks = [n for n in self.lhs.nodes if self.lhs.out_degree(n) == 0]
        
        visited_edges = set()
        self.node_vars = set()
        self.edge_vars = set()
        self.cypher_lines_match = []
        
        for sink in sinks:
            self.traverse_from_sink(sink, visited_edges)

        for line in self.cypher_lines_match:
            print(line)
        
        return self.cypher_lines_match

    
    def get_lhs_carry_vars(self):
        """
        Get the left-hand side carry variables of the rule.
        """
        return self.node_vars, self.edge_vars
    
    
    def traverse_from_sink(self, node, visited_edges):
        G_rev = self.lhs.reverse()
        for pred in G_rev.neighbors(node):
            edge = (pred, node)
            if edge not in visited_edges:
                # Create variable names
                n_var = f"n{pred}"
                m_var = f"m{node}"
                e_var = f"e{pred}{node}"

                self.node_vars.update([n_var, m_var])
                self.edge_vars.add(e_var)

                cypher_line = f"({n_var})-[{e_var}]->({m_var})"
                self.cypher_lines_match.append(cypher_line)

                visited_edges.add(edge)
                self.traverse_from_sink(pred, visited_edges)