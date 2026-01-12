import argparse
import json
import numpy as np


def summarize_nested_sampling(npz_path: str = 'nested_sampling_results.npz', decimals: int = 8) -> dict:
    data = np.load(npz_path, allow_pickle=True)
    if isinstance(data, np.lib.npyio.NpzFile):
        samples = data['samples']
        logl = data['logl'] if 'logl' in data.files else None
    else:
        samples = data
        logl = None

    iterations = len(data['samples'])

    if decimals is not None:
        rounded = np.round(samples.astype(np.float64), decimals=decimals)
        unique_param_sets = int(np.unique(rounded, axis=0).shape[0])
    else:
        unique_param_sets = int(np.unique(samples, axis=0).shape[0])

    return {'iterations': iterations, 'unique_param_sets': unique_param_sets}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Summarize nested sampling results from NPZ file')
    parser.add_argument('--file', '-f', default='nested_sampling_run/nested_sampling_results.npz', help='Path to NPZ results file')
    parser.add_argument('--decimals', type=int, default=6, help='Rounding decimals for uniqueness check')
    args = parser.parse_args()

    summary = summarize_nested_sampling(args.file, args.decimals)
    print(json.dumps(summary, indent=2))
