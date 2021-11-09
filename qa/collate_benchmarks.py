#!/usr/bin/env python3
import os
import re
import sys

import cudf
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

BATCH_FILE_RE = re.compile(r'([0-9]+)\.csv')
SUMMARY_DIR_NAME = 'summary'

def gather_perf_reports(benchmark_dir):
    _, model_dirs, _ = next(os.walk(benchmark_dir))
    for model in model_dirs:
        if model != SUMMARY_DIR_NAME:
            model_dir = os.path.join(benchmark_dir, model)
            for file_ in os.listdir(model_dir):
                file_match = BATCH_FILE_RE.match(file_)
                if file_match:
                    batch = int(file_match.groups()[0])
                    data = cudf.read_csv(os.path.join(model_dir, file_))
                    yield model, batch, data


def collate_raw_data(benchmark_dir):
    all_data = []
    for model, batch, data in gather_perf_reports(benchmark_dir):
        annotations = cudf.DataFrame(
            {
                'Model': [model] * data.shape[0],
                'Batch Size': [batch] * data.shape[0]
            },
            columns=('Model', 'Batch Size')
        )
        all_data.append(cudf.concat([annotations, data], axis=1))
    return cudf.concat(all_data, axis=0, ignore_index=True)


def plot_lat_tp(data):
    all_models = data['Model'].unique().to_pandas()
    plt.xscale('log')
    plt.yscale('log')
    for model in all_models:
        model_data = raw_data.loc[data['Model'] == model].to_pandas()
        plt.scatter(
            model_data['p99 latency'],
            model_data['Inferences/Second']
        )
    plt.title('Throughput vs. Latency (log-log)')
    plt.xlabel('p99 Latency (microseconds)')
    plt.ylabel('Throughput (samples/s)')
    plt.legend(all_models)



if __name__ == '__main__':
    benchmark_dir = sys.argv[1]
    raw_data = collate_raw_data(benchmark_dir)
    summary_dir = os.path.join(benchmark_dir, SUMMARY_DIR_NAME)
    os.makedirs(summary_dir, exist_ok=True)
    raw_data.to_csv(os.path.join(summary_dir, "raw_data.csv"))

    if plt is not None:
        plot_lat_tp(raw_data)
        plt.savefig(os.path.join(summary_dir, 'latency_throughput.png'))
