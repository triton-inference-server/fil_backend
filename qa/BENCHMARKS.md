# FIL Backend Benchmarks
In order to facilitate performance analysis during development of the FIL
backend, the `qa/run_benchmarks.sh` scripts can run a simple set of benchmarks
against standard models. To run this script, first install the benchmarking
conda environment:
```bash
conda env create -f conda/environments/triton_benchmark.yml
```

Next, start the Triton server with the provided benchmark models. Note that you
will need [git lfs](https://git-lfs.github.com/) to checkout these models. You
may start the server by running the following command from the repo root:

```bash
docker run \
  --rm \
  --gpus=all \
  --name benchmark_server \
  -p 8000:8000 \
  -p 8001:8001 \
  -p 8002:8002 \
  -v $PWD/qa/benchmark_repo:/models \
  triton_fil \
  tritonserver \
  --model-repository=/models
```

Here, `triton_fil` is used as the Docker image, since this is the standard tag
used during development, but you may run the benchmarks against any Triton
image which contains the FIL backend.

In a separate terminal, you may now invoke the benchmark script itself as
follows:
```bash
conda activate triton_benchmark
./qa/run_benchmarks.sh
```

The benchmark script will provide output in the `qa/benchmark_output`
directory. Each model tested will have its own directory with `.csv` files
representing results for various batch sizes. The `summary` directory will also
contain a `.csv` collating the data from each run as well as a `.png` showing
throughput vs. p99 latency for all tested models on a single graph.

The benchmark script can be configured using a few different environment
variables, summarized below:
- `MODELS`: A space-separated list of the models to benchmark (defaults to
  standard benchmarking models)
- `BATCHES`: A space-separated list of the batch sizes to use during
  benchmarking (defaults to `'1 16 128 1024'`)
- `MAX_LATENCY`: The maximum latency (in ms) to explore during benchmarking
  (defaults to 5 ms)
