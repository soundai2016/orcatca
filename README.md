# OrcaTCA

OrcaTCA trains killer whale call detectors and measures how detection thresholds transfer across Northeast Pacific recording collections. Negative windows are labelled as humpback whale or background sound.

## Results and model weights

Experimental artifacts and checkpoints are distributed through [Hugging Face: soundai2016/orcatca](https://huggingface.co/soundai2016/orcatca), not GitHub. The release preserves their project-relative paths:

| Path in the Hugging Face repository | Contents |
| --- | --- |
| `result/` | Analyses, figures, features, derived tables and training records |
| `runs/` | Training runs, evaluation outputs and model checkpoints |
| `state/` | Saved experiment state, excluding `cache/` directories |
| `**/best.pt` | Additional checkpoints outside the directories above, if present |
| `model_index.csv` | Checkpoint paths, sizes, seeds and SHA256 hashes |
| `release_manifest.json` | Release file inventory and SHA256 hashes |

Raw DCLDE recordings and training scripts are **not** included in the Hugging Face release. Check the [repository files](https://huggingface.co/soundai2016/orcatca/tree/main) before downloading; the artifact files become available after the release is uploaded.

## Download artifacts for reproduction

From the OrcaTCA source-code root, download the published artifacts into the same relative paths expected by the code:

```bash
cd /data/orcatca
python -m pip install -U huggingface_hub
hf download soundai2016/orcatca \
  --local-dir . \
  --include 'result/**' \
  --include 'runs/**' \
  --include 'state/**' \
  --include '**/best.pt' \
  --include 'best.pt' \
  --include 'model_index.csv' \
  --include 'release_manifest.json'
```

This downloads only published artifacts, leaving the source-code `README.md` untouched. Use a clean checkout or back up any existing `result/`, `runs/` and `state/` directories before downloading, so files from different releases are not mixed. For an exact release, add `--revision <commit-hash>` to the `hf download` command. See `model_index.csv` to locate individual checkpoints and `release_manifest.json` for the file hashes. The `state/cache/` directories are intentionally not published.

To download **model weights only**:

```bash
hf download soundai2016/orcatca \
  --local-dir . \
  --include '**/best.pt' \
  --include 'best.pt' \
  --include 'model_index.csv'
```

## Use downloaded results

These commands reuse saved data and results; they do not start a new training run. Use Python 3.11+:

```bash
cd /data/orcatca
python -m pip install -r requirements.txt
python scripts/acoustic_examples.py
python scripts/plot.py
python scripts/check_figures.py
```

## New training from dataset

For a new experiment, keep the [DCLDE archive](https://doi.org/10.25921/15ey-mh50) at `orcatca/dclde_2027_killer_whales`. On the Ascend training server:

```bash
conda activate orcatca-npu
bash setup.sh
bash run.sh start --fresh --evaluate-holdouts
bash run.sh status
```
