"""Guard against plotting incomplete seed groups or bridging invalid sizes."""
import unittest

from evaluation.plot_timings import select_completed, value


def row(size, seed=0, outcome='complete'):
    result = dict(family='random_clifford', size=str(size), seed=str(seed), outcome=outcome)
    for engine in ('pyzx', 'cypher', 'hybrid'):
        result.update({engine+'_complete':'True', engine+'_active':'True',
                       engine+'_rewrite':'1', engine+'_total':'2'})
    return result


class PlotSelectionTests(unittest.TestCase):
    def test_one_failed_seed_ends_all_engine_curves_before_that_size(self):
        data = [row(64, i) for i in range(3)] + [row(128, i) for i in range(3)] + [row(256)]
        data[4]['hybrid_complete'] = 'False'
        selected, coverage = select_completed(data)
        self.assertEqual([r['size'] for r in selected], ['64']*3)
        self.assertEqual([r['included'] for r in coverage], [True, False, False])

    def test_nonfinite_and_finalization_interrupted_cases_are_not_plotted(self):
        for mutate in (lambda r:r.update(cypher_total='nan'),
                       lambda r:r.update(outcome='finalization_interrupted')):
            invalid = row(128)
            mutate(invalid)
            self.assertEqual(len(select_completed([row(64), invalid])[0]), 1)

    def test_noop_has_runtime_but_no_speedup(self):
        noop = row(128)
        noop['cypher_active'] = 'False'
        self.assertEqual(len(select_completed([noop])[0]), 1)
        self.assertEqual(value(noop, 'cypher', 'rewrite'), 1)
        self.assertIsNone(value(noop, 'cypher', 'rewrite', speedup=True))


if __name__ == '__main__':
    unittest.main()
