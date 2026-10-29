# Paper figures from the finalized campaign

Eight figures cover growing rules, disjoint copies and full-reduce circuits in both rewrite-only and total time, plus circuit speedups for each timing scope. `main.pdf` is the compiled eight-page preview with captions. Each named figure also has a tightly cropped vector PDF and a TikZ snippet.

## Selection and interpretation

Only complete three-engine configurations with three successful measured repetitions per engine are included. Raw input hashes, repetition sets, and medians were checked. Every planned seed at a size must qualify. Each family stops at its first incomplete size; there is no extrapolation or bridging of incomplete sizes. All 210 completed configurations qualify.

These are usable timing measurements, not semantic correctness certificates. Most large-output semantic checks remain inconclusive. The campaign used changing runtime cutoffs, and local-layered seeds repeat the same input. Error bars span configuration medians, not within-run variation or confidence intervals. No-op points remain in runtime figures but have no speedup.

## Maximum plotted workload sizes

| Workload | Maximum size parameter |
|---|---:|
| bialgebra | 128 |
| copies_bialgebra | 4096 |
| copies_copy | 4096 |
| copies_gadget_fusion | 8192 |
| copies_hadamard_cancel | 2048 |
| copies_identity | 8192 |
| copies_lcomp | 8192 |
| copies_pivot | 4096 |
| copies_pivot_boundary | 4096 |
| copies_pivot_gadget | 4096 |
| copies_spider_fusion | 4096 |
| copies_supplementarity | 512 |
| copy | 4096 |
| gadget_fusion | 4096 |
| hadamard_cancel | 4096 |
| identity | 4096 |
| identity_chain | 4096 |
| layered_global | 128 |
| layered_local | 128 |
| lcomp | 1024 |
| pivot | 1024 |
| pivot_boundary | 1024 |
| pivot_gadget | 1024 |
| random_clifford | 128 |
| random_clifford_t | 128 |
| random_t_heavy | 128 |
| spider_fusion | 4096 |
| supplementarity | 4096 |

Rule axes use actual input vertices, recovered from raw observations; the table above uses workload size parameters. Copy axes count disjoint motifs. Circuit axes count qubits and explicitly label 128.

## Use in the paper

For TikZ, load `pgfplots`, the `groupplots` library, and compatibility 1.18, then include the named snippet inside a full-width figure. The PDFs may instead be included with `\includegraphics[width=\textwidth]{circuits-total.pdf}`. Use the matching caption in `main.tex`.

Generate from the repository root with `myenv/Scripts/python.exe evaluation/plot_timings.py`. Compile `main.tex` with `pdflatex` from this directory. To rebuild a cropped PDF, run `pdflatex -jobname=circuits-total circuits-total-standalone.tex` (substitute the figure name).

`plotted-data.csv` is the exact selected data, including input vertex counts. `coverage.csv` includes all planned size groups and exclusion reasons. `plot-manifest.json` records the original finalized CSV and observation hashes. Earlier campaigns are not pooled.
