"""Render the README benchmark figure. Requires matplotlib and numpy."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

HERE = Path(__file__).resolve().parent
data = json.loads((HERE / 'flashprefill-humaneval-32k.json').read_text())
rows = data['summary']
assert len(rows) == 18 and all(r['complete'] for r in rows)
assert sum(r['completed'] for r in rows) == 1368
lookup = {(r['context'], r['mode'], r['alpha']): r for r in rows}
modes = ['baseline', 'mtp3', 'dflash2']
colors = ['#8794A5', '#1769ED']
ink = '#182536'
muted = '#56677C'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 15,
                     'text.color': ink, 'axes.labelcolor': ink,
                     'xtick.color': muted, 'ytick.color': muted})
fig, axes = plt.subplots(3, 3, figsize=(18, 12.8), facecolor='white')
fig.subplots_adjust(left=.095, right=.98, top=.76, bottom=.14,
                    wspace=.16, hspace=.36)
fig.text(.055, .949, 'LSE  /  HumanEval+', fontsize=31, weight='bold')
fig.text(.055, .912, 'Qwen3.8-27B Q4  ·  AMD R9700  ·  HRX / Loom  ·  1,368 completed generations',
         fontsize=17, color=muted)
fig.text(.055, .864, '51–55% faster prefill at 32K with alpha 0.1',
         fontsize=22, weight='bold', color=colors[1])
fig.legend(handles=[Patch(facecolor=colors[0], label='FlashPrefill Off'),
                    Patch(facecolor=colors[1], label='FlashPrefill alpha 0.1')],
           loc='upper right', bbox_to_anchor=(.983, .886), ncol=2,
           frameon=False, fontsize=15, handlelength=1.0, columnspacing=1.5)
contexts = [(0, 'Standard', '164 tasks / configuration'),
            (16384, '16K context', '32 tasks / configuration'),
            (32768, '32K context', '32 tasks / configuration')]
metrics = [('pass_at_1', 'Correctness\npass@1 (%)', 100, 116, [0, 25, 50, 75, 100]),
           ('pps', 'Prompt speed\ntokens / second', 1, 780, [0, 200, 400, 600]),
           ('tps', 'Decode speed\ntokens / second', 1, 100, [0, 25, 50, 75])]
for ci, (context, title, subtitle) in enumerate(contexts):
    bounds = axes[0, ci].get_position()
    mid = (bounds.x0 + bounds.x1) / 2
    fig.text(mid, .811, title, ha='center', fontsize=23, weight='bold')
    fig.text(mid, .784, subtitle, ha='center', fontsize=13, color=muted)
    for ri, (metric, ylabel, factor, ymax, ticks) in enumerate(metrics):
        ax = axes[ri, ci]
        x = np.arange(3)
        for ai, alpha in enumerate([None, .1]):
            values = [lookup[context, mode, alpha][metric] * factor for mode in modes]
            bars = ax.bar(x + (ai - .5) * .43, values, width=.32,
                          color=colors[ai], zorder=3)
            ax.bar_label(bars, labels=[f'{v:.1f}' for v in values],
                         padding=5, fontsize=12, weight='bold', color=ink)
        ax.set_xticks(x, ['Baseline', 'MTP3', 'DFlash2'], fontsize=14)
        ax.set_ylim(0, ymax)
        ax.set_yticks(ticks)
        ax.set_xlim(-.65, 2.65)
        ax.grid(axis='y', color='#E6EBF1', linewidth=.7, zorder=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(axis='both', length=0, pad=7, labelsize=12)
        if ci == 0:
            ax.set_ylabel(ylabel, fontsize=16, labelpad=18)
        else:
            ax.tick_params(axis='y', labelleft=False)
fig.text(.055, .091,
         'Standard = original short prompts, not zero tokens. 16K / 32K use the same fixed 32-task subset with added background.',
         fontsize=13, color=muted)
fig.text(.055, .062,
         'Off and alpha 0.1 match task outcomes at Standard and 32K. At 16K: 32/32 vs 31/32 in every mode (HumanEval/97).',
         fontsize=13, color=muted)
fig.text(.055, .033,
         'Long-context tests are modified HumanEval+. Rates = total tokens / total timed seconds, including JIT. BF16 K/V · batch 1024 · dense decode.',
         fontsize=12, color=muted)
fig.savefig(HERE / 'flashprefill-humaneval-32k.png', dpi=180, facecolor='white')
plt.close(fig)
