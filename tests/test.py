import concurrent.futures
import io
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from queue import Queue, Empty
from time import perf_counter
from uuid import uuid4

import cuml
import numpy as np
import requests
import tritonclient.http as triton_http
import tritonclient.grpc as triton_grpc
import tritonclient.utils.cuda_shared_memory as shm
from tritonclient import utils as triton_utils

GRPC_URL = 'localhost:8001'
HTTP_URL = 'localhost:8000'
MODEL_NAME = 'fil'
MODEL_VERSION = '1'

def get_triton_client(protocol="grpc"):
    try:
        return get_triton_client.cache[protocol]
    except AttributeError:
        get_triton_client.cache = {}
    except KeyError:
        if protocol == 'grpc':
            get_triton_client.cache[
                protocol
            ] = triton_grpc.InferenceServerClient(
                url=GRPC_URL,
                verbose=False
            )
        elif protocol == 'http':
            get_triton_client.cache[
                protocol
            ] = triton_http.InferenceServerClient(
                url=HTTP_URL,
                verbose=False,
                concurrency=12
            )
        else:
            raise RuntimeError('Bad protocol: "{}"'.format(protocol))
    return get_triton_client(protocol=protocol)

def set_up_triton_io(
        client, arr, protocol='grpc', shared_mem=None, predict_proba=False,
        num_classes=1):
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
        # TODO
        triton_input = triton_grpc.InferInput('input__0', arr.shape, 'FP32')
        triton_output = triton_grpc.InferRequestedOutput('output__0')

        input_size = arr.size * arr.itemsize
        output_size = arr.shape[0] * arr.itemsize
        if predict_proba:
            output_size *= num_classes

        input_name = 'input_{}'.format(uuid4().hex)
        output_name = 'output_{}'.format(uuid4().hex)

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

        shared_memory_regions = client.get_cuda_shared_memory_status().regions
        assert input_name in shared_memory_regions
        assert output_name in shared_memory_regions

    return (triton_input, triton_output, output_handle, input_name, output_name)

def get_result(response, output_handle):
    if output_handle is None:
        return response.as_numpy('output__0')
    else:
        network_result = response.get_output('output__0')
        return shm.get_contents_as_numpy(
            output_handle,
            triton_utils.triton_to_np_dtype(network_result.datatype),
            network_result.shape
        )

def triton_predict(
        arr, protocol='grpc', shared_mem=None, predict_proba=False,
        num_classes=1, attempts=3):
    client = get_triton_client(protocol=protocol)
    start = perf_counter()
    triton_input, triton_output, handle, input_name, output_name = set_up_triton_io(
        client,
        arr,
        protocol=protocol,
        shared_mem=shared_mem,
        predict_proba=predict_proba,
        num_classes=num_classes
    )

    result = client.infer(
        MODEL_NAME,
        model_version=MODEL_VERSION,
        inputs=[triton_input],
        outputs=[triton_output]
    )
    output = get_result(result, handle)
    elapsed = perf_counter() - start

    if input_name is not None:
        client.unregister_cuda_shared_memory(name=input_name)
    if output_name is not None:
        client.unregister_cuda_shared_memory(name=output_name)

    return output, elapsed

if __name__ == '__main__':
    total_rows = 131072
    batch_sizes = (128, 1, 1024, 131072)
    # total_rows = 4096*2**0
    # batch_sizes = (128, 1, total_rows)
    shared_mem = (None, 'cuda')
    protocol = 'grpc'
    model_path = '/home/whicks/proj_cuml_triton/test_repository/fil/1/xgboost.model'
    concurrency = 12

    concurrency = max(2, concurrency - concurrency % 2)

    client = get_triton_client(protocol=protocol)
    client.unregister_cuda_shared_memory()

    config = client.get_model_config('fil').config
    model_type = config.parameters.get('model_type', 'xgboost')
    features = config.input[0].dims[0]
    output_dims = config.output[0].dims
    try:
        num_classes = output_dims[1]
        predict_proba = True
    except IndexError:
        num_classes = 1
        predict_proba = False

    fil_model = cuml.ForestInference.load(model_path, output_class=True)

    total_batch = np.random.rand(total_rows, features).astype('float32')

    fil_result = fil_model.predict(total_batch)

    # Warmup
    for _ in range(10):
        triton_predict(total_batch[0:1])

    triton_result, latency = triton_predict(
        total_batch[0:1],
        protocol='grpc',
        shared_mem='cuda',
        predict_proba=predict_proba,
        num_classes=num_classes
    )
    np.testing.assert_almost_equal(triton_result, fil_result[0])

    predict_networked = partial(
        triton_predict,
        protocol=protocol,
        shared_mem=None,
        predict_proba=predict_proba,
        num_classes=num_classes
    )
    predict_shared = partial(
        triton_predict,
        protocol=protocol,
        shared_mem='cuda',
        predict_proba=predict_proba,
        num_classes=num_classes
    )

    queue = Queue()
    results = []

    locks = {}

    def predict_worker():
        locks[threading.get_ident()] = threading.Lock()
        while True:
            try:
                next_input = queue.get()
            except Empty:
                continue
            if next_input is None:
                queue.task_done()
                return
            arr, indices, sharing = next_input
            sharing = 'cuda'
            with locks[threading.get_ident()]:
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
    queue.join()
    total = perf_counter() - start

    for _ in range(concurrency):
        queue.put(None)

    for thread in pool:
        thread.join()

    throughput = len(batch_sizes) * total_rows / total
    per_sample = 1 / throughput

    request_latency = 0
    row_latency = 0
    for indices, (triton_result, batch_latency) in results:
        np.testing.assert_almost_equal(
            triton_result, fil_result[indices[0]: indices[1]]
        )
        request_latency += batch_latency
        row_latency += batch_latency / (indices[1] - indices[0])

    request_latency /= len(results)
    row_latency /= len(results)

    print("*****************************************")
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
    print(f"Model type: {model_type}")
    print(f"Protocol: {protocol}")
    print(f"Concurrency: {concurrency}")
    print("GPUs: 2x Quadro RTX 8000")
    print("CPUs: 12x Intel Xeon Gold 6128 @ 3.40 GHz")
    print("-----------------------------------------")
    print("Throughput, Request latency, Time/sample, Row latency")
    print(f"{throughput}, {request_latency}, {per_sample}, {row_latency}")
    print("*****************************************")
