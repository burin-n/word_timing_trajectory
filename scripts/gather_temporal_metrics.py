"""Combine explicitly supplied temporal score CSVs from a completed batch."""

import argparse
import csv
from pathlib import Path


def gather_scores(input_paths: list[Path], output_path: Path) -> None:
    rows = []
    columns = None
    for path in input_paths:
        with path.open(newline='') as stream:
            reader = csv.DictReader(stream)
            if columns is None:
                columns = reader.fieldnames
            if reader.fieldnames != columns:
                raise ValueError(f'Score columns do not match: {path}')
            rows.extend(reader)
    if not rows:
        raise ValueError('No temporal scores to aggregate')
    rows.sort(key=lambda row: (row['model_name'], int(row['layer_id']),
                               row['label_name'], row['estimator']))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', type=Path, nargs='+')
    parser.add_argument('--output-path', type=Path, required=True)
    args = parser.parse_args()
    gather_scores(args.inputs, args.output_path)


if __name__ == '__main__':
    main()
