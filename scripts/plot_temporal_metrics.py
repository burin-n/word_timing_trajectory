"""Plot layer-wise temporal metrics from an explicitly supplied score CSV."""

import math
import os
import argparse
import matplotlib.pyplot as plt
import pandas as pd

MODEL_TRANSFORMER_LAYERS: list[tuple[str, list[int]]] = [
    ("vibevoice",  list(range(5, 6))),
    ("whisper",    list(range(3, 9))),
    ("mimi",       list(range(1, 9))),
    ("dinosr",     list(range(1, 13))),
    ("spidr",      list(range(1, 13))),
    ("melhubert",  list(range(1, 13))),
    ("hubert",     list(range(1, 13))),
    ("wav2vec2",   list(range(1, 13))),
    ("wavlm",      list(range(1, 13))),
    # ("encodec_24khz", [0]),
    ("spectrogram", [0]),
    # ("spectrogram-layernorm", [0])
]


MODEL_DEFAULT_LAYERS: list[tuple[str, list[int]]] = [
    ("vibevoice", [5]),
    ("whisper",   list(range(2, 9))),
]


def parse_layer_range(s: str) -> list[int]:
    """Parse '3-6' → [3,4,5,6], or '5' → [5]."""
    if "-" in s:
        start, end = s.split("-", 1)
        return list(range(int(start), int(end) + 1))
    return [int(s)]


def get_default_layers(model_name: str) -> list[int] | None:
    name = model_name.lower()
    for pattern, layers in MODEL_DEFAULT_LAYERS:
        if pattern in name:
            return layers
    return None


def get_transformer_layers(model_name: str) -> list[int] | None:
    name = model_name.lower()
    for pattern, layers in MODEL_TRANSFORMER_LAYERS:
        if pattern in name:
            return layers
    return None


def shift_layer_id(model_name: str, layer_id: int) -> int:
    """Renumber layer_id so the first transformer layer = 0."""
    if "spectrogram" in model_name.lower() or "vibevoice" in model_name.lower():
        return layer_id
    layers = get_transformer_layers(model_name)
    if layers is None:
        return layer_id
    return layer_id - min(layers)


def shift_default_layer_id(model_name: str, layer_id: int) -> int:
    """Renumber layer_id so the first default layer = 0."""
    if "spectrogram" in model_name.lower() or "vibevoice" in model_name.lower():
        return layer_id
    layers = get_default_layers(model_name)
    if layers is None:
        return layer_id
    return layer_id - min(layers)


LOWER_IS_BETTER = {
    "radius_cv_train",
    "radius_cv_test",
    "optimal_line_mse_train",
    "optimal_line_mse_test",
}


SLOPE_TARGET = 1 / (2 * math.pi)


MODEL_DISPLAY_NAMES = {
    "hubert-base-ls960":                "HuBERT",
    "melhubert-360h":                   "MelHuBERT",
    "wav2vec2-base":                    "Wav2Vec2",
    "wavlm-base":                       "WavLM",
    "dinosr-reproduced":                "DinoSR",
    "spidr":                            "SpidR",
    "whisper-base":                     "Whisper",
    "mimi":                             "Mimi",
    "VibeVoice-ASR-HF_acoustic":        "VibeVoice-acoustic",
    "VibeVoice-ASR-HF_semantic":        "VibeVoice-semantic",
    "hubert-base-random":               "Unt. HuBERT",
    "whisper-random":                   "Unt. Whisper",
    "spectrogram":                      "Spectrogram",
}


MODEL_ORDER = list(MODEL_DISPLAY_NAMES.values())


METRIC_DISPLAY_NAMES = {
    "train_r2": "R2", "r2_train": "R2",
    "test_r2":  "R2", "r2_test":  "R2",
    # "radius_cv_train": "radius-CV", "radius_cv_test": "radius-CV",
    "optimal_line_mse_train": "RMSE",
    "optimal_line_mse_test": "RMSE",
    "slope": "|slope|",
    "endpoint_completion": "completion%",
    # "endpoint_completion": "rot%",
    "mean_explained_variance_train":  "EV-mean",
    "mean_explained_variance_test":   "EV-mean",
    "mean_explained_variance_subspace_train": "EV-mean-sub",
    "mean_explained_variance_subspace_test":  "EV-mean-sub",
    "frame_explained_variance_train": "EV-sample",
    "frame_explained_variance_test":  "EV-sample",
}


EXTRA_COLS = {
    "train": [
        ("optimal_line_mse_train",         "RMSE"),
        ("slope",                          "slope"),
        ("endpoint_completion",            "completion%"),
        ("mean_explained_variance_train",  "EV-mean"),
        ("mean_explained_variance_subspace_train", "EV-2D/sub"),
        ("frame_explained_variance_train", "EV-sample"),
        # ("radius_cv_train",                "radius-CV"),
    ],
    "test": [
        ("optimal_line_mse_test",          "RMSE"),
        ("mean_explained_variance_test",   "EV-mean"),
        ("mean_explained_variance_subspace_test", "EV-2D/sub"),
        ("frame_explained_variance_test",  "EV-sample"),
        # ("radius_cv_test",                 "radius-CV"),
    ],
}


def apply_display_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    unknown = sorted(set(df["model_name"].unique()) - set(MODEL_DISPLAY_NAMES))
    if unknown:
        print(f"Warning: dropping {len(unknown)} unknown model(s): {unknown}")
    df = df[df["model_name"].isin(MODEL_DISPLAY_NAMES)]
    df["model_name"] = df["model_name"].map(MODEL_DISPLAY_NAMES)
    return df


def _train_counterpart(metric: str) -> str:
    """Return the train-split name for a test metric, or the metric itself if already train."""
    if metric.startswith("test_"):
        return "train_" + metric[5:]
    elif metric.endswith("_test"):
        return metric[:-5] + "_train"
    return metric


def _is_lower_better(col: str) -> bool:
    return _train_counterpart(col) in LOWER_IS_BETTER


def _is_slope_metric(col: str) -> bool:
    return _train_counterpart(col) == "slope"


def _best_layer_idx(sub: pd.DataFrame, col: str):
    if _is_slope_metric(col):
        return (sub[col].abs() - SLOPE_TARGET).abs().idxmin()
    if _is_lower_better(col):
        return sub[col].idxmin()
    return sub[col].idxmax()


def _display_metric_value(col: str, value: float) -> float:
    if col.startswith("optimal_line_mse_"):
        return value ** 0.5
    if col == "slope":
        return abs(value)
    if col == "endpoint_completion":
        return value * 100.0
    return value


EXCLUDED_MODELS = {"VibeVoice-acoustic", "VibeVoice-semantic", "Spectrogram"}
RANDOM_MODELS = {"Unt. HuBERT", "Unt. Whisper"}


def _line_style_for_model(model_name):
    return ":" if model_name in RANDOM_MODELS else "-"

def _marker_style_for_model(model_name):
    return "^" if model_name in RANDOM_MODELS else "o"



def load_results(input_csv):
    """Read only the supplied CSV; never scan neighboring experiment outputs."""
    df = pd.read_csv(input_csv)
    required = {"model_name", "layer_id", "label_name", "estimator"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Required column(s) {missing} not found in {input_csv}")
    if df.empty:
        raise ValueError(f"No temporal scores in {input_csv}")
    layers = pd.to_numeric(df["layer_id"], errors="raise")
    if layers.isna().any() or (layers % 1 != 0).any():
        raise ValueError("layer_id must contain integer layer IDs")
    df["layer_id"] = layers.astype(int)
    return df


def get_args(argv=None):
    parser = argparse.ArgumentParser(description="Layer-wise overlay plot of an exp20 metric across models.")
    parser.add_argument("--input-csv", required=True, help="Score CSV, normally score_all.csv from exp2")
    parser.add_argument("--metric", default=None, help="Column to plot (default: all available train metrics, or test metrics with --testset)")
    parser.add_argument("--label", nargs="+", default=["word"], help="Filter by label_name (default: word). Pass 'all' to include all labels.")
    parser.add_argument("--estimator", nargs="+", default=None, help="Filter by estimator (default: all estimators present)")
    parser.add_argument("--layer", default=None, help="Restrict layer search to a range, e.g. '3-6' (inclusive). Default: all layers.")
    parser.add_argument("--transformer-only", action=argparse.BooleanOptionalAction, default=False,
                        help="Keep only transformer layers per model (default: off).")
    parser.add_argument("--save-dir", default=None, help="Directory to save figures (default: <input CSV directory>/plot_exp20)")
    parser.add_argument("--testset", action="store_true", help="Select available test metrics instead of train metrics")
    return parser.parse_args(argv)


def _plot_one(sub_df, metric, save_path, ylim=None):
    fig, ax = plt.subplots(figsize=(3,2))
    color_id = 0
    for model_name in MODEL_ORDER:
        if model_name in EXCLUDED_MODELS:
            continue
        model_sub = sub_df[sub_df["model_name"] == model_name]
        if model_sub.empty:
            continue
        model_sub = model_sub.sort_values("layer_id")
        if model_sub[metric].isna().all():
            print(f"Warning: no values for {model_name}, metric={metric}; skipping.")
            color_id += 1
            continue
        metric_values = _display_metric_value(metric, model_sub[metric])
        if model_name == "Spectrogram":
            ax.axhline(metric_values.iloc[0], linestyle="--", label=model_name, alpha=0.5, markersize=2, color=f"C{color_id}")
        else:
            line_style = _line_style_for_model(model_name)
            marker_style = _marker_style_for_model(model_name)
            marker_size_standard = 3 if model_name in RANDOM_MODELS else 2
            marker_size_best = 8 if model_name in RANDOM_MODELS else 6

            ax.plot(model_sub["layer_id"], metric_values, linestyle=line_style,
                    marker=marker_style, label=model_name, alpha=0.5, markersize=marker_size_standard, color=f"C{color_id}")
            best_layer_id = _best_layer_idx(model_sub, metric)
            ax.plot(model_sub.loc[best_layer_id, "layer_id"], metric_values.loc[best_layer_id],
                    marker=marker_style, label=model_name, alpha=0.7, markersize=marker_size_best, color=f"C{color_id}")
        color_id += 1

    if not ax.lines:
        plt.close(fig)
        print(f"Warning: no values for metric={metric}; skipping {save_path}.")
        return False

    if metric == "slope":
        ax.axhline(1 / 2 / math.pi, color="black", linestyle=":", linewidth=2, alpha=0.7, zorder=0)

    ax.set_xlabel("layer")
    if "r2" in metric:
        metric_display_name = r"$R^2$"
    else:
        metric_display_name = METRIC_DISPLAY_NAMES.get(metric, metric)
    ax.set_ylabel(metric_display_name)
    if ylim is not None:
        ax.set_ylim(ylim)

    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")
    return True


def _save_legend(sub_df, save_path):
    fig_dummy, ax_dummy = plt.subplots()
    handles, labels = [], []
    color_id = 0
    for model_name in MODEL_ORDER:
        if model_name in EXCLUDED_MODELS:
            continue
        if sub_df[sub_df["model_name"] == model_name].empty:
            continue
        if model_name == "Spectrogram":
            h = ax_dummy.axhline(0, linestyle="--", alpha=0.5, label=model_name, color=f"C{color_id}")
        else:
            marker_size_= 8 if model_name in RANDOM_MODELS else 6
            h, = ax_dummy.plot([], [], linestyle=_line_style_for_model(model_name), marker=_marker_style_for_model(model_name),
                               markersize=marker_size_, alpha=0.5, label=model_name, color=f"C{color_id}")
        handles.append(h)
        labels.append(model_name)
        color_id += 1
    plt.close(fig_dummy)

    # ncols = math.ceil(len(handles) / 3)
    ncols = 5
    fig_leg = plt.figure()
    fig_leg.legend(handles, labels, ncol=ncols, loc="center")
    fig_leg.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig_leg.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig_leg)
    print(f"Saved: {save_path}")


def main(argv=None):
    args = get_args(argv)
    df = load_results(args.input_csv)
    df = apply_display_names(df)

    if args.label != ["all"]:
        df = df[df["label_name"].isin(args.label)]
    if args.estimator is not None:
        df = df[df["estimator"].isin(args.estimator)]
    if args.layer is not None:
        df = df[df["layer_id"].isin(parse_layer_range(args.layer))]
    df = df[~df["model_name"].isin(EXCLUDED_MODELS)]
    if df.empty:
        raise ValueError("No plottable rows match the model, label, estimator, and layer selection")
    if not args.transformer_only:
        parts = []
        for model_name, sub in df.groupby("model_name"):
            layers = get_default_layers(model_name)
            parts.append(sub[sub["layer_id"].isin(layers)] if layers is not None else sub)
        df = pd.concat(parts, ignore_index=True)
        if df.empty:
            raise ValueError("No plottable rows remain after model layer filtering")
        df["layer_id"] = df.apply(
            lambda row: shift_default_layer_id(row["model_name"], row["layer_id"]), axis=1
        )
    if args.transformer_only:
        parts = []
        for model_name, sub in df.groupby("model_name"):
            layers = get_transformer_layers(model_name)
            if layers is None:
                print(f"Warning: no transformer-layer mapping for '{model_name}', keeping all layers")
                parts.append(sub)
            else:
                parts.append(sub[sub["layer_id"].isin(layers)])
        df = pd.concat(parts, ignore_index=True)
        if df.empty:
            raise ValueError("No plottable rows remain after model layer filtering")
        df["layer_id"] = df.apply(
            lambda row: shift_layer_id(row["model_name"], row["layer_id"]), axis=1
        )

    if args.metric is None:
        split = "test" if args.testset else "train"
        r2_candidates = ["test_r2", "r2_test"] if args.testset else ["train_r2", "r2_train"]
        r2_col = next((c for c in r2_candidates if c in df.columns), None)
        extra = [col for col, _ in EXTRA_COLS[split]]
        all_candidates = ([r2_col] if r2_col else []) + extra
        metrics = [m for m in all_candidates if m in df.columns]
    else:
        metrics = [args.metric]
        missing = [m for m in metrics if m not in df.columns]
        if missing:
            raise ValueError(f"Metric(s) {missing} not found in columns: {sorted(df.columns)}")

    if not metrics:
        raise ValueError("No supported metrics found in the input CSV for the selected split")

    save_dir = args.save_dir or os.path.join(os.path.dirname(args.input_csv), "plot_exp20")

    for (estimator, label_name), sub in df.groupby(["estimator", "label_name"]):
        # Keep one color assignment across metrics and the shared legend.
        has_values = sub.groupby("model_name")[metrics].count().sum(axis=1) > 0
        sub = sub[sub["model_name"].isin(has_values[has_values].index)]
        if sub.empty:
            print(f"Warning: no metric values for estimator={estimator}, label={label_name}; skipping.")
            continue
        saved = False
        for metric in metrics:

            if "r2" in metric:
                ylim = [0.5, 1.05]
            else:
                ylim = None

            save_path = os.path.join(save_dir, estimator, label_name, f"plot_layerwise_{metric}.png")
            saved = _plot_one(sub, metric, save_path, ylim=ylim) or saved
        legend_path = os.path.join(save_dir, estimator, label_name, "legend.png")
        if saved:
            _save_legend(sub, legend_path)


if __name__ == "__main__":
    main()
