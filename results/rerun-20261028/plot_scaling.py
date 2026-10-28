"""Render measured circuit scaling from the completed campaign summary."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent / 'performance'
rows = json.loads((root/'family-summary.json').read_text())
families = [('random_clifford', 'Clifford'), ('random_clifford_t', 'Clifford + T'),
            ('random_t_heavy', 'T-heavy'), ('layered_local', 'Local layered'),
            ('layered_global', 'Global layered')]
colors = {'pyzx':'#36454f', 'cypher':'#0072b2', 'hybrid':'#d55e00'}
labels = {'pyzx':'PyZX', 'cypher':'Cypher', 'hybrid':'Cypher + C++'}
for scope in ('rewrite', 'total'):
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), layout='constrained')
    for ax, (family, title) in zip(axes.flat, families):
        selected = sorted([r for r in rows if r['family']==family], key=lambda r:r['size'])
        for engine in colors:
            good = [r for r in selected if r[engine+'_'+scope] is not None]
            ax.plot([r['size'] for r in good], [r[engine+'_'+scope] for r in good],
                    marker='o', linewidth=2, color=colors[engine], label=labels[engine])
        ax.set(xscale='log', yscale='log', title=title, xlabel='Qubits', ylabel='Seconds')
        ax.set_xticks([3, 8, 16, 32, 64, 128], labels=['3','8','16','32','64','128'])
        ax.grid(True, which='both', alpha=.2)
    axes.flat[-1].axis('off')
    handles, names = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles, names, loc='upper left', frameon=False, fontsize=13)
    axes.flat[-1].text(0, .60, 'Three measured repetitions per seed\nThree seed settings per size\nGeometric mean of seed medians\n\nLocal-layered seeds are identical inputs.\nLog scales; finite-size evidence only.\nChecks run outside timers.',
                       transform=axes.flat[-1].transAxes, fontsize=11, va='top', linespacing=1.5)
    fig.suptitle(f'ZXdb benchmark rerun: {scope} time — 2026-10-28', fontsize=16)
    fig.savefig(root/f'circuit-{scope}-scaling.png', dpi=160)
    fig.savefig(root/f'circuit-{scope}-scaling.svg')
    plt.close(fig)
