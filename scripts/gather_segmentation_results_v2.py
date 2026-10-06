"""Gather best parameters or threshold pairs from saved segmentation scores."""

import argparse
from dataclasses import dataclass
from itertools import product
import json
import math
from pathlib import Path
import re

import pandas as pd

# when --mode=paired-threshold, the script returns experiment pairs
# threshold-to-no-threshold selects best experiment using threshold and then find its pair without the threshold
# no-threshold-to-threshold selects best experiment without threshold and then find its pair with the threshold
# pairs are two experiments with identical configurations, except the thershold (on/off).
PAIRED_DIRECTIONS = ('threshold-to-no-threshold', 'no-threshold-to-threshold')
# The script selects the best experiment separately for each combination of
# model, layer, estimator, concept space (testspace), and boundary unit (testunit).
GROUP_COLUMNS = ['model', 'layer_id', 'estimator', 'testspace', 'testunit']
# For csv formatting purposes. In both modes, these columns appear first in the CSV.
PREFIX_COLUMNS = GROUP_COLUMNS + ['tuned_metric']
# Both modes include only these four scores in the CSV, in this order.
# In both modes, --percentage multiplies them by 100 and --decimals rounds them
# after any p ercentage conversion.
METRIC_COLUMNS = ['F1', 'OS', 'R-value', 'TokenF1']
METRIC_ALIASES = {'Over-segmentation': 'OS', 'Token F1': 'TokenF1'}
# The script searches through experimets with identical setups, allowing only `param_id` and `threshold_max_angle` to vary.
# This effectively finds experiments that only differ in `threshold_max_angle` 
#   in order to compare how `threshold_max_angle` impact the score.
PAIR_EXCLUDE_KEYS = {'param_id', 'threshold_max_angle'}
# These configuration columns will be removed from the final CSV.
DROP_COLUMNS = ['sampling_rate', 'subsampling_factor', 'preprocessor']


@dataclass
class Candidate:
    path: Path
    config: dict
    scores: dict[str, float]


def read_config(config_path=None):
    # The first line contains a dictionary, for example:
    # {'prominence': np.float64(0.1), 'preprocessor': None}
    # Returns data={'prominence': 0.1, 'preprocessor': 'none'}.
    with open(config_path) as inf:
        inline = inf.readlines()[0].strip().replace("'", '"')
        while inline != re.sub(r"np\.float64\(([\d.eE+\-]+)\)", r'\g<1>', inline):
            inline = re.sub(r"np\.float64\(([\d.eE+\-]+)\)", r'\g<1>', inline)
        inline = re.sub(r"None", r'"none"', inline)
        data = json.loads(inline)
    return data


def read_score_resfile(path: Path) -> dict[str, float]:
    scores = {}
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        for part in line.split(','):
            name, separator, value = part.partition(':')
            name = name.strip()
            try:
                if not separator or not name:
                    raise ValueError('expected metric: value')
                scores[METRIC_ALIASES.get(name, name)] = float(value)
            except ValueError as error:
                raise ValueError(f'{path}:{lineno}: invalid score {part.strip()!r}: {error}') from error
    return scores


def load_candidates(result_dir: Path, config_dir: Path, config_cache=None) -> list[Candidate]:
    """Read each score once; share configs across boundary units in this run."""
    if config_cache is None:
        config_cache = {}
    candidates = []
    for res_path in sorted(result_dir.glob('*.res'), key=lambda path: path.stem):
        config_path = config_dir / res_path.stem / 'config'
        if config_path not in config_cache:
            config_cache[config_path] = read_config(config_path)
        candidates.append(
            Candidate(
                res_path, 
                config_cache[config_path], 
                read_score_resfile(res_path)
            )
        )
    return candidates

def filter_candidates(candidates: list[Candidate], filters: dict) -> list[Candidate]:
    filtered = []
    for candidate in candidates:
        missing = filters.keys() - candidate.config.keys()
        if missing:
            raise ValueError(f'{candidate.path}: missing config fields required by filters: {sorted(missing)}')
        if all( candidate.config[key] == value for key, value in filters.items()):
            filtered.append(candidate)
    return filtered


def select_best(candidates: list[Candidate], metric: str, *, skip_missing=False) -> Candidate | None:
    """Keep the source score floor and first-on-tie selection."""
    best = None
    best_score = -1e9
    for candidate in candidates:
        score = candidate.scores.get(METRIC_ALIASES.get(metric, metric))
        if score is None:
            if skip_missing:
                continue
            raise ValueError(f'{candidate.path}: missing tuned metric {metric!r}')
        if score > best_score:
            best, best_score = candidate, score
    return best


def make_candidate_pair_key(candidate: Candidate) -> tuple:
    return tuple(sorted( (k, v) for k, v in candidate.config.items() if k not in PAIR_EXCLUDE_KEYS))


def select_pair(candidates: list[Candidate], metric: str, direction: str) -> list[Candidate]:
    if direction not in PAIRED_DIRECTIONS:
        raise ValueError(f'Unsupported paired_direction: {direction}')

    # store scores for experiments using thresholds
    thresholded = {}
    # store scores for experiments using no thresholds
    no_threshold = {}
    
    for candidate in candidates:
        max_angle = candidate.config.get('threshold_max_angle', math.nan)
        
        try:
            is_pi = abs(max_angle - math.pi) <= 1e-12
        except TypeError as error:
            raise ValueError(f'{candidate.path}: threshold_max_angle must be numeric for pairing') from error
        
        groups = no_threshold if is_pi else thresholded
        # assigns group to candidates based on their configurations
        # candidates with the same configurations are in the same group.
        candidate_key = make_candidate_pair_key(candidate)
        groups.setdefault(candidate_key, []).append(candidate)
    
    if direction == 'threshold-to-no-threshold':
        selected_groups, paired_groups = thresholded, no_threshold
    else: # 'no-threshold-to-threshold'
        selected_groups, paired_groups = no_threshold, thresholded

    # Preserve source group insertion order, including ties across match keys.
    pool = [candidate for group in selected_groups.values() for candidate in group]
    best_candidate = select_best(pool, metric, skip_missing=True)
    if best_candidate is None:
        return []

    # find the pair of the best candidate
    paired = select_best(
        paired_groups.get(make_candidate_pair_key(best_candidate), []), 
        metric, skip_missing=True
    )
    return [best_candidate, paired] if paired is not None else [best_candidate]


def make_rows(group: dict, metric: str, best_candidate: list[Candidate]) -> list[dict]:
    rows = []
    for candidate in best_candidate:
        scores = candidate.scores
        scores = {name: scores[name] for name in METRIC_COLUMNS if name in scores}
        row = {**group, 'tuned_metric': metric, **scores, **candidate.config}
        rows.append(row)
    return rows


def format_results(rows: list[dict], *, percentage=False, decimals=None) -> pd.DataFrame:
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    # Sort experiment groups while keeping rows within each pair in their original order.
    result = result.sort_values(PREFIX_COLUMNS, kind='stable')

    result = result.drop(columns=DROP_COLUMNS, errors='ignore')
    
    first = PREFIX_COLUMNS + [name for name in METRIC_COLUMNS if name in result]
    result = result[first + [name for name in result if name not in first]]
    
    for name in METRIC_COLUMNS:
        if name in result:
            if percentage:
                result[name] = result[name] * 100
            if decimals is not None:
                result[name] = result[name].round(decimals)
    return result


def gather_results(
    prefix, *, mode='best', testset='dev',
    models=('hubert-base-ls960',), estimators=('Z_CPCA', 'LEACE'), layer_ids=('09',),
    testspaces=('word', 'syllable', 'phone'), testunits=('word', 'syllable', 'phone'),
    tuned_metrics=('R-value',), n_components=2, preprocessor='none',
    use_absolute_threshold=False, paired_direction='threshold-to-no-threshold',
    percentage=False, decimals=None, verbose=False, eval_folder_suffix='eval',
) -> pd.DataFrame:
    if mode not in ('best', 'paired-threshold'):
        raise ValueError(f'Unsupported mode: {mode}')
    if mode == 'paired-threshold' and paired_direction not in PAIRED_DIRECTIONS:
        raise ValueError(f'Unsupported paired_direction: {paired_direction}')

    filters = {}
    if preprocessor is not None:
        filters['preprocessor'] = preprocessor
    if n_components:
        filters['n_components'] = n_components
    if mode == 'best' and not use_absolute_threshold:
        # Preserve the source's special prominence-baseline threshold.
        filters['threshold_max_angle'] = 1 if 'prominence' in estimators else math.pi
    if verbose:
        print('config filter:', filters)

    rows, config_cache = [], {}
    for model, estimator, layer, space, unit in product(models, estimators, layer_ids, testspaces, testunits):
        base = Path(prefix) / estimator / model / layer
        config_dir = base / f'{testset}_{space}-space'
        result_dir = base / f'{testset}_{space}-space_{eval_folder_suffix}' / f'{unit}-boundary'
        candidates = load_candidates(result_dir, config_dir, config_cache)
        candidates = filter_candidates(candidates, filters)
        
        if verbose:
            print('base_path:', result_dir)
            print('n_matching_configs:', len(candidates))
        
        group = dict(zip(GROUP_COLUMNS, (model, layer, estimator, space, unit)))
        for metric in tuned_metrics:
            if mode == 'best':
                best_ = select_best(candidates, metric)
                if best_ is not None:
                    missing = set(METRIC_COLUMNS) - best_.scores.keys()
                    if missing:
                        raise ValueError(f'{best_.path}: missing report metrics: {sorted(missing)}')
                best_candidate = [] if best_ is None else [best_]
            else:
                best_candidate = select_pair(candidates, metric, paired_direction)

            rows.extend(make_rows(group, metric, best_candidate))
            if verbose:
                print(f'  {metric}:', [candidate.path.stem for candidate in best_candidate])
    return format_results(rows, percentage=percentage, decimals=decimals)


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', default='best', choices=['best', 'paired-threshold'])
    parser.add_argument('--paired-direction', default=PAIRED_DIRECTIONS[0], choices=PAIRED_DIRECTIONS)
    parser.add_argument('--segmentation-dir', type=Path, required=True, help='Root containing saved scores and configs')
    parser.add_argument('--testset', default='dev', choices=['dev', 'test'])
    parser.add_argument('--n-components', type=int, default=2)
    parser.add_argument('--preprocessor', default='none', choices=['none', 'normalize', 'hypersphere'])
    parser.add_argument('--models', nargs='+', default=['hubert-base-ls960'])
    parser.add_argument('--estimators', nargs='+', default=['Z_CPCA', 'LEACE', 'LDA'])
    parser.add_argument('--layer-ids', nargs='+', default=['09'])
    parser.add_argument('--testspaces', nargs='+', default=['full', 'word', 'syllable', 'phone'])
    parser.add_argument('--testunits', nargs='+', default=['word', 'syllable', 'phone'])
    parser.add_argument('--tuned-metrics', nargs='+', default=['R-value'])
    parser.add_argument('--eval-folder-suffix', default='eval',
                        help='Suffix appended to <testset>_<testspace>-space for score folders')
    parser.add_argument('--use-absolute-threshold', action='store_true',
                        help='Best mode: allow all threshold_max_angle values')
    parser.add_argument('--percentage', action='store_true', help='Multiply report metrics by 100')
    parser.add_argument('--decimals', type=int, default=None, help='Round after percentage conversion')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--save-csv', type=Path, default=None)
    return parser


def main() -> None:
    options = vars(argument_parser().parse_args())
    prefix = options.pop('segmentation_dir')
    output = options.pop('save_csv')
    result = gather_results(prefix, **options)
    if result.empty:
        print('No matching result files found.')
    print(result.to_string())
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output, index=False)
        print(f'\nSaved to {output}')


if __name__ == '__main__':
    main()
