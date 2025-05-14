from query import Query


class QueryConstructor:
    
    def __init__(self, rule):
        self.rule = rule
        self.query = Query()

    def build_query(self):
        """
        Every query consists of a match and then a series of rewrites that translates the matching graph into the new graph.
        """
        
        self.query.add_match(self.rule.get_lhs_pattern(), self.rule.get_lhs_carry_vars())
        