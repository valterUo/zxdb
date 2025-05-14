class Query:
    
    def __init__(self):
        self.match = None
        self.carry_on_vars = {}
        self.redirect_incoming = None
        self.redirect_outgoing = None
        self.create_node = None
        self.cascade = None
        self.delete_old = None
        self.full_query = []

    def add_match(self, matching_pattern, lhs_carry_vars):
        """
        Add a match to the query.
        """
        self.keyword = "MATCH "
        self.matching_pattern = matching_pattern
        self.lhs_carry_vars_nodes, self.lhs_carry_vars_edges = lhs_carry_vars
        self.match = f"{self.keyword} " + ", ".join(self.matching_pattern) + "\n" + \
            f"WITH {', '.join(self.lhs_carry_vars_nodes)}, {', '.join(self.lhs_carry_vars_edges)}"
        
        self.full_query.append(self.match)
        
        #return self.match, self.lhs_carry_vars_nodes, self.lhs_carry_vars_edges
        self.carry_on_vars["nodes"] = self.lhs_carry_vars_nodes
        self.carry_on_vars["edges"] = self.lhs_carry_vars_edges
        self.carry_on_vars["old_nodes"] = "old_nodes"
        self.carry_on_vars["old_edges"] = "old_edges"
        
    def add_constraint(self, constraint):
        """
        Add a constraint to the query.
        """
        self.full_query.append(f"WHERE {constraint}")


    def add_cascade(self, distinct = False):
        """
        Add a cascade to the query.
        """
        
        if distinct:
            label = "DISTINCT "
        else:
            label = ""
        
        all_carry_vars = []
        for v in self.carry_on_vars:
            all_carry_vars.extend(self.carry_on_vars[v])
        all_carry_vars = list(set(all_carry_vars))
            
        self.cascade = f"WITH {label} " + ", ".join(all_carry_vars) + "\n"

        self.full_query.append(self.cascade)


    def add_create_node(self):
        new_node_var = "new_node"
        
        self.create_node = f"""
            CREATE ({new_node_var}:Vertex)
            """.format(new_node=new_node_var)
            
        self.carry_on_vars["new_node"] = new_node_var
        self.full_query.append(self.create_node)


    def add_redirect_incoming_edges_to_single_node(self):
        """
        Add a redirect for incoming edges to the new node.
        """
        try:
            new_node_var = self.carry_on_vars["new_node"]
            old_nodes_var = self.carry_on_vars["old_nodes"]
        except KeyError as e:
            raise KeyError(f"Missing key in carry_on_vars: {e}")

        self.redirect_incoming = f"""
            WITH {new_node_var}, {old_nodes_var}
            UNWIND {old_nodes_var} AS old
            OPTIONAL MATCH (src)-[r_in]->(old)
            WHERE NOT src IN {old_nodes_var}
            WITH DISTINCT {new_node_var}, src, r_in, {old_nodes_var}
            WHERE src IS NOT NULL
            CREATE (src)-[new_in:LEG_TO]->({new_node_var})
            SET new_in = r_in
            DELETE r_in
            """.format(new_node=new_node_var, old_nodes=old_nodes_var)
        
        self.full_query.append(self.redirect_incoming)
    
    
    def add_redirect_outgoing_edges_to_single_node(self):
        """
        Add a redirect for outgoing edges from the new node.
        """
        try:
            new_node_var = self.carry_on_vars["new_node"]
            old_nodes_var = self.carry_on_vars["old_nodes"]
        except KeyError as e:
            raise KeyError(f"Missing key in carry_on_vars: {e}")

        self.redirect_outgoing = f"""
            WITH {new_node_var}, {old_nodes_var}
            UNWIND {old_nodes_var} AS old
            OPTIONAL MATCH (old)-[r_out]->(dst)
            WHERE NOT dst IN {old_nodes_var}
            WITH DISTINCT {new_node_var}, dst, r_out, {old_nodes_var}
            WHERE dst IS NOT NULL
            CREATE ({new_node_var})-[new_out:LEG_TO]->(dst)
            SET new_out = r_out
            DELETE r_out
            """.format(new_node=new_node_var, old_nodes=old_nodes_var)
        
        self.full_query.append(self.redirect_outgoing)


    def delete_old_nodes(self):
        """
        Delete the old nodes.
        """
        try:
            old_nodes_var = self.carry_on_vars["old_nodes"]
        except KeyError as e:
            raise KeyError(f"Missing key in carry_on_vars: {e}")

        self.delete_old = f"""
            WITH {old_nodes_var}
            UNWIND {old_nodes_var} AS old
            DETACH DELETE old
            """.format(old_nodes=old_nodes_var)
        
        self.full_query.append(self.delete_old)
        
    
    def get_full_query(self):
        """
        Get the full query.
        """
        return "\n".join(self.full_query)
            
    