#!/usr/bin/env python3
import os
import re
import sys

import cudf
import numpy as np

from scipy.spatial import ConvexHull

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

def pts_to_line(pt1, pt2):
    slope = (pt2[1] - pt1[1]) / (pt2[0] - pt1[0])
    intercept = pt1[1] - slope * pt1[0]
    return (slope, intercept)

def scatter_to_hull(pts):
    hull = ConvexHull(pts)
    pts = pts[hull.vertices]
    pts = pts[pts[:, 0].argsort(), :]
    slope, intercept = pts_to_line(pts[0, :], pts[-1, :])
    filtered_pts = pts[pts[:, 1] >= slope * pts[:, 0] + intercept]
    return np.concatenate((pts[(0,), :], filtered_pts, pts[(-1,), :]))


def plot_lat_tp(data, latency_percentile=99):
    all_models = data['Model'].unique().to_pandas()
    plt.xscale('log')
    plt.yscale('log')
    for model in all_models:
        model_data = raw_data.loc[data['Model'] == model].to_pandas()
        hull = scatter_to_hull(model_data[
            [f'p{latency_percentile} latency', 'Inferences/Second']
        ].values)
        plt.plot(hull[:, 0], hull[:, 1], '-', label=model)
    plt.title('Throughput vs. Latency (log-log)')
    plt.xlabel('p99 Latency (microseconds)')
    plt.ylabel('Throughput (samples/s)')
    plt.legend(all_models)

def plot_througput(data, budget, output_dir):
    filtered_data = data[data['p99 latency'] <= budget][
        ['Model', 'Inferences/Second']
    ]
    maximums = filtered_data.groupby('Model').max()
    maximums.sort_index(inplace=True)

    budget_ms = round(budget / 1000)

    raw_data.to_csv(os.path.join(output_dir, f'{budget_ms}.csv'))

    if plt is not None:
        plt.bar(
            maximums.index.values_host,
            maximums['Inferences/Second'].values_host
        )
        plt.xticks(rotation=90)
        plt.title(
            f'Throughput for p99 latency budget of {budget_ms} ms'
        )
        plt.subplots_adjust(bottom=0.35)
        plt.savefig(os.path.join(output_dir, f'{budget_ms}.png'))
        plt.close()


if __name__ == '__main__':
    benchmark_dir = sys.argv[1]
    raw_data = collate_raw_data(benchmark_dir)
    summary_dir = os.path.join(benchmark_dir, SUMMARY_DIR_NAME)
    throughput_dir = os.path.join(summary_dir, 'throughput')
    os.makedirs(throughput_dir, exist_ok=True)
    raw_data.to_csv(os.path.join(summary_dir, "raw_data.csv"))

    try:
        latency_cutoff = float(os.environ['MAX_LATENCY'])
        raw_data = raw_data[
            raw_data['p99 latency'] <= (latency_cutoff * 1000)
        ]
    except KeyError:
        pass  # No latency cutoff specified

    raw_data.to_csv(os.path.join(summary_dir, "filtered_data.csv"))

    plot_througput(raw_data, 1000, throughput_dir)
    plot_througput(raw_data, 5000, throughput_dir)
    plot_througput(raw_data, 20000, throughput_dir)

    if plt is not None:
        plot_lat_tp(raw_data)
        plt.savefig(os.path.join(summary_dir, 'latency_throughput.png'))
