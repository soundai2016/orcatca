# Check and reproduce the transfer analysis

Run from `/data/orcatca` after installing `requirements.txt`.

## Generate the figures

```bash
python scripts/plot.py
python scripts/check_figures.py
```

Figures 1 and 2 are written directly to `result/` in PDF and PNG. Their numerical tables and input hashes are in `result/figure_data/derived/`. The default analysis is `result/analysis/transfer_review`; specifying this same path explicitly gives the same output. Run `python scripts/check_label_mixtures.py` to regenerate both mixture tables.

The renderer prints its own path, the selected acoustic features and each output path. The check requires all four top-level outputs to match the recorded code, inputs and features; it exits with status 1 if a file is absent or stale. Files in the older `result/figures/` directory are reported separately and never used as replacements. Install the update in the existing project root, then redraw; simply unzipping a `sourcecode/` folder beneath that root does not update the outer scripts.

Fig. 1 compares the killer whale target with humpback whale and background windows from the same training recording. Fig. 2 pairs the target window’s amplitude envelope with its average spectrum. The acoustic examples illustrate the labels and do not enter the performance estimates.

To recompute the acoustic features from the native recording and draw both figures:

```bash
python scripts/acoustic_examples.py
python scripts/plot.py
```

The extractor checks raw file and annotation hashes. It reuses identical features and backs up the exact previous default set before replacing it. With no explicit input path, the renderer uses current local features or the bundled set. Unknown or edited files are not silently replaced. The standard pair stops if verified acoustic features are unavailable. Use `--acoustic-dir PATH` for an explicit selection. For the fixed analysis, `--stats-only` writes to `result/stats_only/`; single-figure renders use `result/single_figures/` by default. Neither replaces the standard pair.

## Check the numerical results

```bash
python scripts/check_analysis_inputs.py --layout
python scripts/audit_results.py --analysis result/analysis/transfer_review \
  --replay-check result/training_records/analysis_replay_check.json
```

Add `--tables-dir /path/to/tex-tables` to compare LaTeX cells directly with the numerical results.

To write a separate 16-second listening comparison (KW, HW, AB; 0.5-second gaps), run `python scripts/acoustic_examples.py --listening-copy`. This writes `result/acoustic_comparison.wav` with a single gain for all three clips. Brackets in Fig. 1 mark the original annotation times. Selection used training-window spectra rather than detector scores; the 126 screened windows are listed in the full results under `result/validation/acoustic_window_screen.csv`.

Acoustic examples: SIMRES, Tekteksen; DCLDE 2027, NOAA NCEI, [10.25921/15ey-mh50](https://doi.org/10.25921/15ey-mh50). The excerpts and derived displays are attributed under the [archive license text](https://storage.googleapis.com/noaa-passive-bioacoustic/dclde/2027/dclde_2027_killer_whales/CC-BY-4.0.txt), Attribution-ShareAlike 4.0. The source file hash, sample frames and annotation intervals are in the feature metadata.

## Verify the migrated training records

```bash
python scripts/check_analysis_inputs.py --layout --verify
python scripts/extract_training_records.py --verify-runs runs --records result/training_records
```

The first command checks live run files against the recorded SHA-256 read trace. It compares each available `holdout.json` with the same fit in the retained tables: model and fold identities, full-calibration thresholds, window counts, FPR, recall and site-mean pAUC. The 18 primary summaries are in `runs/semantic230_20261029T160625Z_9e1b11a8/`; the 18 binary and fine-auxiliary summaries are in `result/analysis/reference/holdouts/`. The second command checks the live training records against `result/training_records/`. Add `--checkpoint-root runs` to check weights and measure the local file sizes.

In the historical label table, `errors` counts windows flagged as KW: these are false positives for HW/AB and true positives for KW. Use `target_tp / target_positive` for KW recall.

## Recompute from window scores

The migrated `semantic230` directory supplies coarse auxiliary and three-class predictions. The binary and fine-auxiliary prediction streams do not appear in the supplied listings. Their 18 saved holdout summaries and the YAML in `state/requests/` cannot replace them. Set `ORCA_REFERENCE` to the actual path of the original `jel220_20261026T021531Z_ac09b0f3_report.zip`. The primary report ZIP is unnecessary.

```bash
(
  set -e
  : "${ORCA_REFERENCE:?Set ORCA_REFERENCE to the original jel220 report path}"
  primary=runs/semantic230_20261029T160625Z_9e1b11a8
  run=result/analysis/transfer_replay_$(date -u +%Y%m%dT%H%M%SZ)
  PYTHONPATH=src python -m orcatca analyze --source "$primary" --reference "$ORCA_REFERENCE" \
    --expect-trace result/analysis/transfer_review/analysis.json --output "$run" \
    --seed 20261029 --repeats 2000 --crossfit-repeats 5
  python scripts/audit_results.py --analysis "$run"
  python scripts/plot.py --analysis "$run" --figure both \
    --output "$run/figures" --figure-data "$run/figure_data/derived"
  python scripts/check_label_mixtures.py --analysis "$run" \
    --output "$run/figure_data/derived/mixture_bounds.csv"
)
```

The output directory must be new. Input identity is checked across both suites, including the frozen protocol and every score stream. Analysis reads neither audio nor model weights.

If you have the separate prediction companion instead, restore it once to a new directory and set the reference for the same command block:

```bash
python scripts/study_reports.py restore --bundle prediction_inputs.zip --output result/replay_inputs
export ORCA_REFERENCE="$PWD/result/replay_inputs/reference_selected.zip"
```

Without a direct primary suite, replace `primary=...` with `primary=result/replay_inputs/primary_selected.zip` or the path to its original report. For a full 36-fit training-resource export from the two original report ZIPs use `python scripts/extract_training_records.py --reports /path/to/reports --output NEW_DIR`; the prediction companion does not include training logs.
