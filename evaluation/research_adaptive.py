"""Predeclared ascending ladders and checkpoint-derived stopping decisions."""
import random


def series_key(spec):
    # Circuit depth grows at fixed width; rule motif/fan count grows otherwise.
    if spec['family'].startswith(('random_', 'layered_')):
        return spec['family'], spec['size']
    return spec['family'], spec.get('depth')


def scale_value(spec):
    return spec['depth'] if spec['family'].startswith(('random_', 'layered_')) else spec['size']


def ordered_cases(plan):
    if not plan.get('adaptive'):
        order = list(range(len(plan['cases'])))
        random.Random(plan['order_seed']).shuffle(order)
        return order
    groups = {}
    for i, spec in enumerate(plan['cases']):
        groups.setdefault(series_key(spec), []).append(i)
    rng = random.Random(plan['order_seed'])
    keys = list(groups); rng.shuffle(keys)
    for indices in groups.values():
        rng.shuffle(indices)  # Randomize seeds at a fixed scale.
        indices.sort(key=lambda i: scale_value(plan['cases'][i]))
    # Round-robin ladders: one expensive family cannot consume the whole budget.
    return [groups[key][rung] for rung in range(max(map(len, groups.values()), default=0))
            for key in keys if rung < len(groups[key])]


def timed_out(history, repetitions, stop_on='both'):
    if any(r.get('kind') == 'limit_stop' and r.get('reason') == 'both_engines_timeout' for r in history):
        return True
    trials = [r for r in history if r.get('kind') == 'trial' and not r['warmup']]
    completed = {r['trial'] for r in trials}
    if completed == set(range(repetitions)):
        def hit(row):
            statuses = [row['engines'][e].get('status') in ('timeout', 'not_run_after_timeout') for e in ('db', 'pyzx')]
            return any(statuses) if stop_on == 'either' else all(statuses)
        return bool(trials) and all(hit(r) for r in trials)
    # A killed block, crash or setup timeout does not establish that BOTH
    # simplifiers exceeded their budgets. Do not infer an algorithmic limit.
    return False


def is_timeout(exc):
    error = (type(exc).__name__ + ' ' + str(exc)).lower()
    return any(word in error for word in ('timeout', 'timed out', 'deadline exceeded', 'time limit'))


def stop_evidence(plan, spec, histories):
    policy = plan.get('adaptive')
    if not policy:
        return []
    streak = []
    for i in ordered_cases(plan):
        earlier = plan['cases'][i]
        if earlier['id'] == spec['id']:
            break
        if series_key(earlier) != series_key(spec):
            continue
        history = histories.get(earlier['id'], [])
        if not history:
            # An unrun predecessor is not evidence that the series can advance.
            streak = []
        elif timed_out(history, plan['repetitions'], policy.get('stop_on', 'both')):
            streak.append(earlier['id'])
            if len(streak) >= policy['consecutive_timeouts']:
                return streak
        else:
            streak = []
    return []


def blocked_predecessor(plan, spec, histories):
    """No performance inference from a case killed before both engines finish."""
    if not plan.get('adaptive'):
        return []
    for i in ordered_cases(plan):
        earlier = plan['cases'][i]
        if earlier['id'] == spec['id']:
            break
        if series_key(earlier) == series_key(spec) and any(
                r.get('kind') == 'case_budget' for r in histories.get(earlier['id'], [])):
            return [earlier['id']]
    return []
