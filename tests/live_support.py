"""Shared setup for opt-in database regressions."""
import os
import uuid
import unittest

class ResearchLiveTests(unittest.TestCase):
    def setUp(self):
        from neo4j import GraphDatabase
        from zxdb.zxdb import ZXdb
        uri = os.environ.get('ZXDB_URI', 'bolt://localhost:7687')
        auth = (os.environ.get('ZXDB_USER', ''), os.environ.get('ZXDB_PASSWORD', ''))
        with GraphDatabase.driver(uri, auth=auth) as driver:
            with driver.session() as s:
                if s.run('MATCH(n) RETURN count(n) AS n').single()['n']:
                    self.skipTest('Use an empty dedicated database')
        self.db = ZXdb(uri=uri, user=auth[0], password=auth[1])
        self.gid = 'research_test_' + uuid.uuid4().hex

    def tearDown(self):
        self.db.empty_graphdb(self.gid); self.db.close()

