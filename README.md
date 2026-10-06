# Word timing trajectory

This repository hosts code for the paper, "Deep Speech Representations Track Word Timing along a Low-Dimensional Circular Trajectory".

## How to use 

### Input preparation

This repository runs temporal subspace estimation and analysis from existing feature caches. Prepare the feature caches separately; the scripts select frames from cached features using the supplied alignments.

#### Alignments

Word alignments for LibriSpeech dev-clean and test-clean are provided in `alignments/Librispeech`. These alignments specify the word boundaries used to select cached frames.

For segmentation subspace estimation, `exp3a_save_temporal_subspace.sh` currently uses `train-clean-100` and expects `train-clean-100-word.ali`. The bundled training alignment is named `train-clean-100sampled_10h_utt-word.ali`. Before running this stage, supply the expected alignment or edit the script's dataset and alignment paths to use the intended training subset and matching cache.


#### Features

Each dataset cache folder must contain `<layer_id>.npz` files and `manifest.toml`.

Each `<layer_id>.npz` is a NumPy archive containing arrays keyed by utterance ID: `{utt_id: feature_array}`. Each array must have shape `(T, D)`, where `T` is the number of frames and `D` is the feature dimension. Every utterance ID requested by an alignment must exist in the corresponding archive. Features can come from different models, provided the manifest describes their frame timing correctly.

```
FEATURE_CACHE/
└── <model_name>/                 # e.g. hubert-base-ls960
    └── Librispeech/
        ├── dev-clean/
        │   ├── <layer_id>.npz    # e.g. 10.npz
        │   └── manifest.toml
        ├── test-clean/           # same file layout
        └── train-clean-100/      # for segmentation fitting
```

The `manifest.toml` file describes feature information. This example shows manifest.toml for features extracted by HuBERT. 

```
$ cat manifest.toml 
model_name = "hubert-base-ls960"
audio_sampling_rate = 16000
feature_subsampling_factor = 320
feature_frame_rate = 50.0
```


### Run the scripts

1. Use Python **3.13 or later** and install the repository from its root: `python -m pip install .`.
2. Configure `experiments/temporal_trajectory/run_all_exp.sh`: set `FEATURE_DIR` to your cache root and `BASE_OUTPUT` to your output directory. Review the model and layer selections in its child scripts. The current trajectory experiments use `hubert-base-ls960` and `hubert-base-random`, layers 0–12, with dev-clean and test-clean caches. Segmentation uses the same two models at layer 9, with train-clean-100, dev-clean, and test-clean caches.
3. Complete the segmentation evaluation setup below.
4. Run from the repository root:

   ```bash
   bash experiments/temporal_trajectory/run_all_exp.sh
   ```

The example pipeline estimates temporal subspaces, plots mean trajectories, evaluates temporal metrics, and tunes and evaluates word segmentation. It requires the input caches and training alignment configuration described above, plus the external evaluator setup described below; installation alone is not sufficient. Individual stages can also be run using the `exp*.sh` scripts in the same directory.

### Segmentation evaluation setup

Segmentation evaluation uses an external `boundary_eval.py`, which is not included in this repository. See the source and setup notes in `experiments/temporal_trajectory/exp3c_evaluate_segmentation.sh`. Those notes describe an evaluator environment tested with Python 3.10 and `praat-textgrids`, `numpy`, and `tqdm`; use a separate environment from the Python 3.13+ analysis environment.

Set `EVALUATION_SCRIPT` and the evaluator's `PYTHON_BIN` in `run_all_exp.sh` to the script and interpreter you have installed. Its example paths are `Simon_scripts/evaluation/boundary_eval.py` and `Simon_scripts/evaluation/venv/bin/python`, relative to the repository root.

Unpack the bundled evaluation reference archives:

```bash
tar -xzf alignments/Librispeech-Simon-flatten/dev-clean-flatten.tar.gz -C alignments/Librispeech-Simon-flatten
tar -xzf alignments/Librispeech-Simon-flatten/test-clean-flatten.tar.gz -C alignments/Librispeech-Simon-flatten
```

Set `REFERENCE_DIR` to `alignments/Librispeech-Simon-flatten`, containing the extracted `dev-clean-flatten/` and `test-clean-flatten/` folders. These TextGrid references are separate from the `.ali` files used for subspace fitting and temporal analysis.
