"""Independent oracle and benchmark-contract checks; no database needed."""
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pyzx as zx

from evaluate import benchmark_cases, reduce_database, summarize, zero_cases, trial
from evaluation.core import reference, verify, certify_unitary_rewrite
from evaluation.oracle import _quimb_tensor
from evaluation.workloads import RULES, build


class EvaluationTests(unittest.TestCase):
    def test_circuit_grid_uses_requested_sizes_and_seeds(self):
        cases = list(benchmark_cases(True, full_reduce_only=True, qubits=[2, 3],
                                    seeds=2, depth_factor=7))
        self.assertEqual(len(cases), 20)
        self.assertEqual({rule for rule, _, _ in cases}, {'full_reduce'})
        self.assertEqual({spec['size'] for _, spec, _ in cases}, {2, 3})
        self.assertEqual({spec['seed'] for _, spec, _ in cases}, {0, 1})
        for _, spec, _ in cases:
            self.assertEqual(spec['depth'], 2 if spec['family'].startswith('layered') else 7*spec['size'])

    def test_native_full_reduce_dispatch_does_not_use_cypher_control(self):
        db = Mock()
        reduce_database(db, 'full_reduce', 'example', 'native')
        db.full_reduce_native.assert_called_once_with('example')
        db.full_reduce.assert_not_called()
        db.full_reduce_optimized.assert_not_called()

    def test_unitary_identity_certificate_and_counterexamples(self):
        g = build(dict(family='random_clifford_t', size=2, seed=0, depth=12))
        reduced = g.copy(); reference(reduced, 'full_reduce', time.monotonic()+10)
        self.assertEqual(certify_unitary_rewrite(g, reduced, 20)['status'], 'verified')
        wrong = g.copy()
        v = next(v for v in wrong.vertices() if wrong.type(v) == zx.VertexType.Z)
        wrong.add_to_phase(v, .25)
        self.assertNotEqual(certify_unitary_rewrite(g, wrong, 20)['status'], 'verified')
        swapped = g.copy(); swapped.set_outputs(tuple(reversed(swapped.outputs())))
        self.assertNotEqual(certify_unitary_rewrite(g, swapped, 20)['status'], 'verified')
        zero = g.copy(); zero.scalar.is_zero = True
        self.assertEqual(certify_unitary_rewrite(g, zero, 20)['status'], 'failed')
        self.assertEqual(certify_unitary_rewrite(g, g, .001)['status'], 'inconclusive')

    def test_certificate_cache_and_watchdog_exclude_verification(self):
        g = build(dict(family='identity', size=1, seed=0))
        events = []
        db = Mock(); db.driver.submissions = 0
        def certificate(*args):
            events.append('proof')
            return dict(status='verified', reason='test')
        with patch('evaluate.load'), patch('evaluate.read', return_value=g.copy()), \
             patch('evaluate.reference'), patch('evaluate.reduce_database'), \
             patch('evaluate.verify', return_value=dict(status='inconclusive')), \
             patch('evaluate.certify_unitary_rewrite', side_effect=certificate) as proof:
            row = trial(db, g, 'full_reduce', 'test', 'native', ('db', 'pyzx'),
                        unitary_input=True, certificate_timeout=20, certificate_cache={},
                        after_rewrite=lambda: events.append('stop_watchdog'))
        self.assertEqual(events, ['stop_watchdog', 'proof'])
        self.assertEqual(proof.call_count, 1)
        self.assertEqual(row['status'], 'verified')

    def test_tensor_engine_agrees_with_pyzx_on_both_colors_and_zero_arity(self):
        for color in (zx.VertexType.Z, zx.VertexType.X):
            for phase in (0, .25, .5, 1):
                for degree in range(5):
                    g = zx.Graph()
                    v = g.add_vertex(color, phase=phase, row=1)
                    boundaries = [g.add_vertex(zx.VertexType.BOUNDARY, row=2) for _ in range(degree)]
                    for b in boundaries:
                        g.add_edge((v, b), zx.EdgeType.HADAMARD)
                    g.set_outputs(tuple(boundaries))
                    with self.subTest(color=color, phase=phase, degree=degree):
                        np.testing.assert_allclose(_quimb_tensor(g), g.to_tensor(), atol=1e-10)

    def test_zero_maps_and_wrong_phases_cannot_pass(self):
        wire = build(dict(family='identity', size=1, seed=0))
        for name, zero in zero_cases():
            self.assertEqual(verify(wire, zero)['status'], 'failed', name)
            self.assertEqual(verify(zero, zero)['status'], 'verified', name)
        changed = wire.copy()
        changed.set_phase(next(v for v in changed.vertices() if changed.type(v) == 1), .25)
        self.assertEqual(verify(wire, changed)['status'], 'failed')
        tiny = wire.copy(); tiny.scalar.power2 = -10000
        self.assertEqual(verify(wire, tiny)['status'], 'verified')

    def test_native_reference_does_not_time_fingerprinting_or_extra_full_reduce(self):
        for rule, (_, api) in RULES.items():
            batch = rule in ('pivot_gadget', 'pivot_boundary', 'gadget_fusion')
            fn = Mock(side_effect=[True, False] if batch else [None])
            with patch.object(zx.simplify, api, fn):
                reference(zx.Graph(), rule, time.monotonic()+10)
            self.assertEqual(fn.call_count, 2 if batch else 1)

    def test_tiny_nonzero_diagram_is_not_reported_as_a_zero_map_failure(self):
        original = build(dict(family='gadget_fusion', size=128, seed=0))
        result = original.copy()
        reference(result, 'gadget_fusion', time.monotonic()+10)
        self.assertEqual(verify(original, result)['status'], 'inconclusive')

    def test_nonfinite_contraction_is_inconclusive(self):
        g = build(dict(family='identity', size=1, seed=0))
        with patch('evaluation.core.tensor', return_value=np.array([np.inf])):
            self.assertEqual(verify(g, g)['status'], 'inconclusive')

    def test_errors_missing_trials_and_noops_never_become_speedups(self):
        base = dict(stage='benchmark', case='test', rule='pivot', variant='optimized',
                    repetitions=2, spec={}, status='verified', db_changed=True,
                    pyzx_changed=True, db_rewrite_s=1, pyzx_rewrite_s=2,
                    db_total_s=4, pyzx_total_s=2)
        self.assertIsNone(summarize([base])[0]['rewrite_speedup'])
        self.assertEqual(summarize([base, base])[0]['rewrite_speedup'], 2)
        self.assertEqual(summarize([base, base])[0]['total_speedup'], .5)
        for change in ({'status': 'error'}, {'status': 'failed'}, {'db_changed': False}):
            self.assertIsNone(summarize([base, dict(base, **change)])[0]['rewrite_speedup'])
        failed_warmup = dict(base, warmup=True, status='error')
        self.assertFalse(summarize([failed_warmup])[0]['complete'])
        self.assertIsNone(summarize([failed_warmup, base, base])[0]['rewrite_speedup'])
        self.assertIsNone(summarize([dict(base, attempt=0), dict(base, attempt=1)])[0]['rewrite_speedup'])


if __name__ == '__main__':
    unittest.main()
