"""Indexed, graph-scoped candidate frontiers for batched pivot and lcomp.

Only neighbours of a rewritten centre/pair can acquire changed phases or incident
wires. Other candidates retain eligibility: these rules change no surviving
vertex colors and only toggle Hadamard wires. Rebuild when another rule runs.
"""

LABELS = {'pivot': '_ZXdbPivotWork', 'lcomp': '_ZXdbLcompWork'}


def query_for(code, kind):
    code = code.replace('\r', '')
    label = LABELS[kind]
    if kind == 'lcomp':
        code = code.replace('MATCH (center:Node {graph_id: $graph_id, t: 1})',
            f'MATCH (pending:{label} {{graph_id: $graph_id}})\n'
            'WITH collect(pending) AS frontier\nUNWIND frontier AS center\n'
            f'REMOVE center:{label}\nWITH center')
        code = code.replace('WITH collect({center: center, neighbors: neighbors,',
            f'SET center:{label}\nWITH collect({{center: center, neighbors: neighbors,')
        code = code.replace('DETACH DELETE center',
            f'FOREACH (n IN neighbors | SET n:{label})\nDETACH DELETE center')
    else:
        start = 'MATCH (a:Node {graph_id: $graph_id, t: 1})'
        code = code.replace(start,
            f'MATCH (pending:{label} {{graph_id: $graph_id}})\n'
            'WITH collect(pending) AS frontier\nUNWIND frontier AS a\n'
            f'REMOVE a:{label}\nWITH a\n{start}')
        # Either endpoint may have just become eligible. Canonicalize AFTER
        # matching from the frontier, rather than discarding a dirty high-id end.
        code = code.replace('WHERE id(a) < id(b)', 'WHERE id(a) <> id(b)')
        code = code.replace('WITH a, b, id(pe) AS rank',
            'WITH DISTINCT CASE WHEN id(a) < id(b) THEN a ELSE b END AS a,\n'
            '              CASE WHEN id(a) < id(b) THEN b ELSE a END AS b, pe\n'
            f'SET a:{label}, b:{label}\nWITH a, b, id(pe) AS rank')
        code = code.replace('DETACH DELETE a, b',
            f'FOREACH (n IN only_a + only_b + shared | SET n:{label})\nDETACH DELETE a, b')
    return code


def fixpoint(tx, code, kind, key, graph_id):
    """One global seed, indexed frontiers thereafter; cleanup is always scoped."""
    label = LABELS[kind]
    clear = f'MATCH(n:{label} {{graph_id:$graph_id}}) REMOVE n:{label}'
    phase = ('n.phase IN [0.5, 1.5]' if kind == 'lcomp' else
             'n.phase IS NOT NULL AND n.phase % 2 = round(n.phase % 2)')
    # Remove markers left by a previous interrupted invocation, never trust them.
    tx.run(clear, graph_id=graph_id).consume()
    total = 0
    failed = False
    try:
        seeded = tx.run('MATCH(n:Node {graph_id:$graph_id,t:1}) WHERE '+phase+
                        f' SET n:{label} RETURN count(n) AS n', graph_id=graph_id).single()['n']
        if not seeded:
            return 0
        query = query_for(code, kind)
        while True:
            record = tx.run(query, graph_id=graph_id).single()
            changed = record[key] if record else 0
            total += changed
            if not changed:
                return total
    except BaseException:
        failed = True
        raise
    finally:
        # Materialized frontiers and query failures may leave reserved labels.
        # No user properties, relationship identities or foreign graphs change.
        try:
            tx.run(clear, graph_id=graph_id).consume()
        except Exception:
            # Preserve the primary failure if the server/deadline also prevents
            # cleanup. The next invocation discards any stale owned markers.
            if not failed:
                raise
