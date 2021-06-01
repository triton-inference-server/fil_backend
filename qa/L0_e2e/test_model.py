import argparse
import os
import threading
import traceback
from functools import partial
from queue import Queue, Empty
from time import perf_counter, time
from uuid import uuid4

import cuml
import numpy as np
import tritonclient.http as triton_http
import tritonclient.grpc as triton_grpc
import tritonclient.utils.cuda_shared_memory as shm
from tritonclient import utils as triton_utils


STANDARD_PORTS = {
    'http': 8000,
    'grpc': 8001
}


class TritonMessage:
    """Adapter to read output from both GRPC and HTTP responses"""
    def __init__(self, message):
        self.message = message

    def __getattr__(self, attr):
        try:
            return getattr(self.message, attr)
        except AttributeError:
            try:
                return self.message[attr]
            except Exception:  # Re-raise AttributeError
                pass
            raise


def get_triton_client(
        protocol="grpc",
        host='localhost',
        port=None,
        concurrency=4):
    """Get Triton client instance of desired type """

    if protocol == 'grpc':
        if port is None:
            port = 8001
        client = triton_grpc.InferenceServerClient(
            url=f'{host}:{port}',
            verbose=False
        )
    elif protocol == 'http':
        if port is None:
            port = 8000
        client = triton_http.InferenceServerClient(
            url=f'{host}:{port}',
            verbose=False,
            concurrency=concurrency
        )
    else:
        raise RuntimeError('Bad protocol: "{}"'.format(protocol))

    return client


def set_up_triton_io(
        client, arr, protocol='grpc', shared_mem=None, predict_proba=False,
        num_classes=1):
    """Set up Triton input and output objects"""
    if protocol == 'grpc':
        triton_input = triton_grpc.InferInput('input__0', arr.shape, 'FP32')
        triton_output = triton_grpc.InferRequestedOutput('output__0')
    elif protocol == 'http':
        triton_input = triton_http.InferInput('input__0', arr.shape, 'FP32')
        triton_output = triton_http.InferRequestedOutput(
            'output__0',
            binary_data=True
        )

    output_handle = None

    if shared_mem is None:
        if protocol == 'grpc':
            triton_input.set_data_from_numpy(arr)
        elif protocol == 'http':
            triton_input.set_data_from_numpy(arr, binary_data=True)
        else:
            raise RuntimeError('Bad protocol: "{}"'.format(protocol))
        input_name = None
        output_name = None
    elif shared_mem == 'cuda':
        input_size = arr.size * arr.itemsize
        output_size = arr.shape[0] * arr.itemsize
        if predict_proba:
            output_size *= num_classes

        request_uuid = uuid4().hex
        input_name = 'input_{}'.format(request_uuid)
        output_name = 'output_{}'.format(request_uuid)

        output_handle = shm.create_shared_memory_region(
            output_name, output_size, 0
        )

        client.register_cuda_shared_memory(
            output_name, shm.get_raw_handle(output_handle), 0, output_size
        )

        input_handle = shm.create_shared_memory_region(
            input_name, input_size, 0
        )

        shm.set_shared_memory_region(input_handle, [arr])

        client.register_cuda_shared_memory(
            input_name, shm.get_raw_handle(input_handle), 0, input_size
        )

        triton_input.set_shared_memory(input_name, input_size)

        triton_output.set_shared_memory(output_name, output_size)

        shared_memory_regions = client.get_cuda_shared_memory_status()
        try:
            shared_memory_regions = shared_memory_regions.regions
        except AttributeError:  # HTTP response
            shared_memory_regions = [
                region['name'] for region in shared_memory_regions
            ]
        assert input_name in shared_memory_regions
        assert output_name in shared_memory_regions

    return (triton_input, triton_output, output_handle, input_name, output_name)


def get_result(response, output_handle):
    """Convert Triton response to NumPy array"""
    if output_handle is None:
        return response.as_numpy('output__0')
    else:
        network_result = TritonMessage(response.get_output('output__0'))
        return shm.get_contents_as_numpy(
            output_handle,
            triton_utils.triton_to_np_dtype(network_result.datatype),
            network_result.shape
        )


def triton_predict(
        model_name, arr, model_version='1', protocol='grpc', host='localhost',
        port=None, shared_mem=None, predict_proba=False, num_classes=1,
        attempts=3):
    """Perform prediction on a numpy array using a Triton model"""
    if port is None:
        port = STANDARD_PORTS[protocol]
    client = get_triton_client(protocol=protocol, host=host, port=port)
    start = perf_counter()
    try:
        triton_input, triton_output, handle, input_name, output_name = set_up_triton_io(
            client,
            arr,
            protocol=protocol,
            shared_mem=shared_mem,
            predict_proba=predict_proba,
            num_classes=num_classes
        )

        result = client.infer(
            model_name,
            model_version=str(model_version),
            inputs=[triton_input],
            outputs=[triton_output]
        )
    except triton_utils.InferenceServerException as exc:
        # Workaround for Triton Python client bug
        if exc.status() == 'StatusCode.NOT_FOUND' and attempts > 1:
            return triton_predict(
                model_name,
                arr,
                model_version=model_version,
                protocol=protocol,
                host=host,
                port=port,
                shared_mem=shared_mem,
                predict_proba=predict_proba,
                num_classes=num_classes,
                attempts=attempts - 1
            )
        raise
    output = get_result(result, handle)
    elapsed = perf_counter() - start

    if input_name is not None:
        client.unregister_cuda_shared_memory(name=input_name)
    if output_name is not None:
        client.unregister_cuda_shared_memory(name=output_name)

    return output, elapsed


def run_test(
        model_repo=None,
        model_name='xgboost_classification_xgboost',
        model_version=1,
        protocol='grpc',
        total_rows=8192,
        batch_sizes=None,
        shared_mem=(None, 'cuda'),
        concurrency=8,
        timeout=60,
        retries=3,
        host='localhost',
        http_port=STANDARD_PORTS['http'],
        grpc_port=STANDARD_PORTS['grpc']):

    if batch_sizes is None:
        batch_sizes = []
        if total_rows > 128:
            batch_sizes.append(128)
        # Note: We default to putting the batch size 1 runs in between the
        # batch size 128 and 1024 runs in order to increase the likelihood that
        # the server will have to deal with different-sized arrays in a single
        # server-side batch.
        batch_sizes.append(1)
        if total_rows > 1024:
            batch_sizes.append(1024)
        if total_rows > max(batch_sizes):
            batch_sizes.append(total_rows)

    if model_repo is None:
        model_repo = os.path.join(
            os.path.dirname(__file__),
            'model_repository'
        )

    concurrency = max(
        len(shared_mem), concurrency - concurrency % len(shared_mem)
    )

    client = get_triton_client(
        protocol='grpc',
        host=host,
        port=grpc_port
    )

    # Wait up to 60 seconds for server startup
    server_wait_start = time()
    while time() - server_wait_start < 60:
        try:
            if client.is_server_ready():
                break
        except triton_utils.InferenceServerException:
            pass

    client.unregister_cuda_shared_memory()
    client.unregister_system_shared_memory()

    config = client.get_model_config(model_name).config
    features = config.input[0].dims[0]
    output_dims = config.output[0].dims

    predict_proba = config.parameters.get('predict_proba', None)
    if predict_proba is not None and predict_proba.string_value == 'true':
        predict_proba = True
        num_classes = output_dims[0]
    else:
        predict_proba = False
        num_classes = 2

    output_class = config.parameters.get('output_class')
    output_class = (
        output_class is None or output_class.string_value == 'true'
    )

    model_format = config.parameters.get('model_type', None)
    if model_format is None:
        model_format = 'xgboost'
    else:
        model_format = model_format.string_value

    model_path = os.path.join(model_repo, model_name, str(model_version))
    if model_format == 'xgboost':
        model_path = os.path.join(model_path, 'xgboost.model')
        model_type = 'xgboost'
    elif model_format == 'xgboost_json':
        model_path = os.path.join(model_path, 'xgboost.json')
        model_type = 'xgboost_json'
    elif model_format == 'lightgbm':
        model_path = os.path.join(model_path, 'model.txt')
        model_type = 'lightgbm'
    elif model_format == 'treelite_checkpoint':
        model_path = os.path.join(model_path, 'checkpoint.tl')
        model_type = 'treelite_checkpoint'
    else:
        raise RuntimeError('Model format not recognized')

    if model_format == 'treelite_checkpoint':
        pass  # TODO
    else:
        fil_model = cuml.ForestInference.load(
            model_path, output_class=output_class, model_type=model_type
        )

    total_batch = np.random.rand(total_rows, features).astype('float32')

    if predict_proba:
        fil_result = fil_model.predict_proba(total_batch)
    else:
        fil_result = fil_model.predict(total_batch)

    # Perform single-inference tests
    triton_result, _ = triton_predict(
        model_name,
        total_batch[0: 1],
        model_version=model_version,
        protocol='http',
        host=host,
        port=http_port,
        shared_mem=None,
        predict_proba=predict_proba,
        num_classes=num_classes
    )
    np.testing.assert_almost_equal(triton_result, fil_result[0: 1])

    triton_result, _ = triton_predict(
        model_name,
        total_batch[0: 1],
        model_version=model_version,
        protocol='grpc',
        host=host,
        port=grpc_port,
        shared_mem='cuda',
        predict_proba=predict_proba,
        num_classes=num_classes
    )
    np.testing.assert_almost_equal(triton_result, fil_result[0: 1])

    # Perform multi-threaded tests
    def predict_networked(arr):
        try:
            return triton_predict(
                model_name,
                arr,
                model_version=model_version,
                protocol=protocol,
                host=host,
                port=[http_port, grpc_port][protocol == 'grpc'],
                shared_mem=None,
                predict_proba=predict_proba,
                num_classes=num_classes,
                attempts=retries
            )
        except Exception:
            return (None, traceback.format_exc())

    def predict_shared(arr):
        try:
            return triton_predict(
                model_name,
                arr,
                model_version=model_version,
                protocol=protocol,
                host=host,
                port=[http_port, grpc_port][protocol == 'grpc'],
                shared_mem='cuda',
                predict_proba=predict_proba,
                num_classes=num_classes,
                attempts=retries
            )
        except Exception:
            return (None, traceback.format_exc())

    queue = Queue()
    results = []

    def predict_worker():
        while True:
            try:
                next_input = queue.get()
            except Empty:
                continue
            if next_input is None:
                queue.task_done()
                return
            arr, indices, sharing = next_input
            if sharing is None:
                results.append(
                    (indices, predict_networked(arr))
                )
            elif sharing == 'cuda':
                results.append(
                    (indices, predict_shared(arr))
                )
            queue.task_done()

    pool = [
        threading.Thread(target=predict_worker) for _ in range(concurrency)
    ]
    for thread in pool:
        thread.daemon = True
        thread.start()

    start = perf_counter()
    for batch_size_ in batch_sizes:
        for i in range(
            total_rows // batch_size_ +
            int(bool(total_rows % batch_size_))
        ):
            indices = (
                i * batch_size_,
                min((i + 1) * batch_size_, total_rows)
            )

            arr = total_batch[indices[0]: indices[1]]
            queue.put((arr, indices, shared_mem[i % len(shared_mem)]))
    for _ in range(concurrency):
        queue.put(None)

    for thread in pool:
        thread.join(timeout=timeout)
        if thread.is_alive():
            raise RuntimeError("Test run exceeded timeout")
    total = perf_counter() - start

    throughput = len(batch_sizes) * total_rows / total
    per_sample = 1 / throughput

    request_latency = 0
    row_latency = 0
    for indices, (triton_result, batch_latency) in results:
        if triton_result is None:
            # On failure, the second output from the prediction call is the
            # traceback for the exception (stored in the "batch_latency"
            # variable during unpacking)
            error_msg = batch_latency
            raise RuntimeError(
                f'Prediction failed with error:\n\n{error_msg}'
            )
        np.testing.assert_almost_equal(
            triton_result, fil_result[indices[0]: indices[1]]
        )
        request_latency += batch_latency
        row_latency += batch_latency / (indices[1] - indices[0])

    request_latency /= len(results)
    row_latency /= len(results)

    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print(f"Total rows: {total_rows}")
    print("Batch sizes: {}".format(", ".join(
        (str(size_) for size_ in batch_sizes)
    )))
    print("Shared memory: {}".format(", ".join(
        (str(mem) for mem in shared_mem)
    )))
    print(f"Features: {features}")
    print(f"Classes: {num_classes}")
    print(f"Proba: {predict_proba}")
    print(f"Model format: {model_format}")
    print(f"Protocol: {protocol}")
    print(f"Concurrency: {concurrency}")
    print("-----------------------------------------")
    print("Throughput, Request latency, Time/sample, Row latency")
    print(f"{throughput}, {request_latency}, {per_sample}, {row_latency}")
    print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")


def parse_args():
    """Parse CLI arguments for model testing"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--repo',
        help='path to model repository',
        default=None
    )
    parser.add_argument(
        '--host',
        help='URL of Triton server',
        default='localhost'
    )
    parser.add_argument(
        '--http_port',
        type=int,
        help='HTTP port for Triton server',
        default=STANDARD_PORTS['http']
    )
    parser.add_argument(
        '--grpc_port',
        type=int,
        help='GRPC port for Triton server',
        default=STANDARD_PORTS['grpc']
    )
    parser.add_argument(
        '--name',
        help='name for model',
        default='xgboost_classification_xgboost'
    )
    parser.add_argument(
        '--model_version',
        default='1',
        help='model version to be tested',
    )
    parser.add_argument(
        '--protocol',
        choices=('grpc', 'http'),
        default='grpc',
        help='network protocol to use in tests',
    )
    parser.add_argument(
        '--samples',
        type=int,
        help='number of total test samples per batch size',
        default=8192
    )
    parser.add_argument(
        '-b',
        '--batch_size',
        type=int,
        default=[],
        nargs='*',
        help='batch size(s) to use (may be repeated); default based on total'
        ' samples',
        action='append'
    )
    parser.add_argument(
        '--shared_mem',
        choices=('None', 'cuda'),
        default=[],
        nargs='*',
        help='shared memory mode(s) to use (may be repeated)',
        action='append'
    )
    parser.add_argument(
        '--concurrency',
        type=int,
        default=8,
        help='concurrent threads to use for making requests',
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=60,
        help='time in seconds to wait for processing of all samples',
    )
    parser.add_argument(
        '--retries',
        type=int,
        default=3,
        help='allowed retries for network failures',
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    batch_sizes = [
        size_ for inner in args.batch_size for size_ in inner
    ] or None
    shared_mem = [
        [mem, None][mem.lower() == 'none']
        for inner in args.shared_mem for mem in inner
    ] or (None, 'cuda')
    run_test(
        model_repo=args.repo,
        model_name=args.name,
        model_version=args.model_version,
        protocol=args.protocol,
        total_rows=args.samples,
        batch_sizes=batch_sizes,
        shared_mem=shared_mem,
        concurrency=args.concurrency,
        timeout=args.timeout,
        retries=args.retries
    )
