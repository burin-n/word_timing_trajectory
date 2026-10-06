"""Export selected configs from an exp3d results CSV as prediction YAML."""

import argparse
import csv
import math
from pathlib import Path

import yaml

from gather_segmentation_results_v2 import read_config


THRESHOLD_MODES = ('all', 'threshold', 'nothreshold')


def export_parameters(
    input_csv: Path,
    segmentation_dir: Path,
    output_yaml: Path,
    *,
    layer_id: int,
    testset: str = 'dev',
    test_unit: str = 'word',
    tuned_metric: str = 'R-value',
    threshold_mode: str = 'all',
) -> dict:
    """Recover full configs for one layer/unit/metric selection and write YAML."""
    if threshold_mode not in THRESHOLD_MODES:
        raise ValueError(f'unsupported threshold mode: {threshold_mode}')
    configs = {}
    with input_csv.open(newline='') as stream:
        for row in csv.DictReader(stream):
            if (row['testunit'] != test_unit or row['tuned_metric'] != tuned_metric
                    or int(row['layer_id']) != layer_id):
                continue

            model = row['model']
            estimator = row['estimator']
            unit_space = f"{row['testspace']}_space"
            # CSV strings retain the actual layer directory name, including zeroes.
            config_path = (segmentation_dir / estimator / model / row['layer_id']
                           / f"{testset}_{row['testspace']}-space" / row['param_id'] / 'config')
            config = read_config(config_path)
            if threshold_mode != 'all':
                # Use original values and the gatherer's pi tolerance, not CSV rounding
                # or row order (which changes with the selected pairing direction).
                try:
                    is_pi = abs(config.get('threshold_max_angle', math.nan) - math.pi) <= 1e-12
                except TypeError as exc:
                    raise ValueError(f'{config_path}: threshold_max_angle must be numeric') from exc
                if is_pi != (threshold_mode == 'nothreshold'):
                    continue

            config_entry = configs.setdefault(model, {}).setdefault(estimator, {})
            if unit_space in config_entry:
                raise ValueError(f'duplicate YAML entry for {model}/{estimator}/{unit_space}')
            config_entry[unit_space] = {**config, 'test_unit': test_unit}

    if not configs:
        raise ValueError(f'no matching results for layer {layer_id}, '
                         f'boundary unit {test_unit}, metric {tuned_metric}, '
                         f'threshold mode {threshold_mode}')

    # Load and serialize everything before touching an existing output file.
    content = yaml.safe_dump(configs, sort_keys=False)
    output_yaml.parent.mkdir(parents=True, exist_ok=True)
    output_yaml.write_text(content)
    return configs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-csv', type=Path, required=True)
    parser.add_argument('--segmentation-dir', type=Path, required=True)
    parser.add_argument('--output-yaml', type=Path, required=True)
    parser.add_argument('--layer-id', type=int, required=True)
    parser.add_argument('--testset', default='dev')
    parser.add_argument('--test-unit', default='word')
    parser.add_argument('--tuned-metric', default='R-value')
    parser.add_argument('--threshold-mode', choices=THRESHOLD_MODES, default='all',
                        help='Export one side of a paired report, or all selected rows.')
    args = parser.parse_args()
    try:
        export_parameters(
            args.input_csv, args.segmentation_dir, args.output_yaml, layer_id=args.layer_id,
            testset=args.testset, test_unit=args.test_unit, tuned_metric=args.tuned_metric,
            threshold_mode=args.threshold_mode,
        )
    except KeyError as exc:
        parser.error(f'missing CSV column: {exc.args[0]}')
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f'Wrote selected parameters to {args.output_yaml}')


if __name__ == '__main__':
    main()
