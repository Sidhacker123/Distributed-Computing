import math
import numpy as np
try:
    import cupy as cp
except ImportError as e:
    raise ImportError("CuPy is required for GPU computations. Please install CuPy with CUDA support.") from e
try:
    from dask.distributed import Client, wait
except ImportError as e:
    raise ImportError("Dask is required for distributed computations. Please install dask[distributed].") from e
try:
    from dask_cuda import LocalCUDACluster
except ImportError:
    LocalCUDACluster = None

# Cache for compiled CUDA kernel
_mm_kernel = None

def available_gpus():
    """Return the number of available CUDA GPUs."""
    try:
        count = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError:
        count = 0
    return count

def get_matrix_multiply_kernel():
    """Load/compile the CUDA matrixMultiplyShared kernel and return a RawKernel object."""
    global _mm_kernel
    if _mm_kernel is None:
        kernel_path = __package__ + "/kernels/matrix_multiply.cu" if __package__ else "kernels/matrix_multiply.cu"
        try:
            with open(kernel_path, 'r') as f:
                code = f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"CUDA kernel source not found at {kernel_path}")
        _mm_kernel = cp.RawKernel(code, 'matrixMultiplyShared')
    return _mm_kernel

def run_matrix_multiply_kernel(A, B, measure_time=False):
    """
    Perform matrix multiplication C = A * B on a GPU using the custom CUDA kernel.
    A and B can be NumPy or CuPy arrays. They will be transferred to GPU (as float32) if not already.
    Returns the result as a CuPy array (and kernel execution time in milliseconds if measure_time=True).
    """
    # Ensure inputs are on GPU as float32 CuPy arrays
    A_gpu = cp.array(A) if isinstance(A, np.ndarray) else A
    B_gpu = cp.array(B) if isinstance(B, np.ndarray) else B
    if A_gpu.dtype != cp.float32:
        A_gpu = A_gpu.astype(cp.float32)
    if B_gpu.dtype != cp.float32:
        B_gpu = B_gpu.astype(cp.float32)
    M, K = A_gpu.shape
    K2, N = B_gpu.shape
    assert K == K2, "Inner dimensions must agree for multiplication"
    # Allocate output array on GPU
    C_gpu = cp.empty((M, N), dtype=cp.float32)
    kernel = get_matrix_multiply_kernel()
    TILE = 16  # must match TILE_SIZE in CUDA kernel
    grid_x = (N + TILE - 1) // TILE
    grid_y = (M + TILE - 1) // TILE
    if measure_time:
        start_evt = cp.cuda.Event()
        end_evt = cp.cuda.Event()
        start_evt.record()
    kernel((grid_x, grid_y), (TILE, TILE, 1), (A_gpu, B_gpu, C_gpu, M, N, K))
    if measure_time:
        end_evt.record()
        end_evt.synchronize()
        elapsed_ms = cp.cuda.get_elapsed_time(start_evt, end_evt)
    else:
        cp.cuda.Stream.null.synchronize()
    return (C_gpu, elapsed_ms) if measure_time else C_gpu

def partition_matrix(matrix, num_parts, axis=0):
    """Split a NumPy array into `num_parts` chunks along the given axis."""
    if num_parts <= 1:
        return [matrix]
    return np.array_split(matrix, num_parts, axis=axis)

def start_cluster(num_workers):
    """
    Launch a local Dask cluster with one worker per GPU. Returns (cluster, client).
    Requires dask_cuda to be installed.
    """
    if num_workers < 1:
        raise ValueError("Number of workers must be at least 1")
    if LocalCUDACluster is None:
        raise ImportError("dask_cuda is not installed. Cannot start CUDA cluster.")
    avail = available_gpus()
    if avail == 0:
        raise RuntimeError("No GPUs available for cluster.")
    if num_workers > avail:
        print(f"Warning: Requested {num_workers} GPUs, but only {avail} available. Using {avail}.")
        num_workers = avail
    cluster = LocalCUDACluster(n_workers=num_workers, threads_per_worker=1, CUDA_VISIBLE_DEVICES=list(range(num_workers)))
    client = Client(cluster)
    return cluster, client

def execute_distributed_multiply(A, B, client, partition='row'):
    """
    Perform distributed matrix multiplication C = A * B using Dask across the client's workers.
    A and B should be NumPy arrays on the driver. Returns the result as a NumPy array.
    """
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, "Inner dimensions must agree for multiplication"
    if partition == 'row':
        # Row-wise partitioning: split A into chunks by rows, broadcast B
        num_workers = len(client.scheduler_info()['workers'])
        A_chunks = partition_matrix(A, num_workers, axis=0)
        A_futures = client.scatter(A_chunks, broadcast=False)
        B_future = client.scatter(B, broadcast=True)
        tasks = [client.submit(run_matrix_multiply_kernel, a_chunk, B_future) for a_chunk in A_futures]
        results = client.gather(tasks)
        C = np.vstack(results)
        return C
    elif partition == 'block':
        # 2D block partitioning: split A into row blocks and B into column blocks
        num_workers = len(client.scheduler_info()['workers'])
        row_parts = int(math.floor(math.sqrt(num_workers)))
        col_parts = int(math.ceil(num_workers / row_parts))
        A_chunks = partition_matrix(A, row_parts, axis=0)
        B_chunks = partition_matrix(B, col_parts, axis=1)
        A_futures = client.scatter(A_chunks, broadcast=False)
        B_futures = client.scatter(B_chunks, broadcast=False)
        tasks = []
        for A_part in A_futures:
            for B_part in B_futures:
                tasks.append(client.submit(run_matrix_multiply_kernel, A_part, B_part))
        results = client.gather(tasks)
        # Assemble the result matrix from block results
        C = np.empty((M, N), dtype=np.float32)
        row_sizes = [chunk.shape[0] for chunk in A_chunks]
        col_sizes = [chunk.shape[1] for chunk in B_chunks]
        idx = 0
        row_offset = 0
        for r_size in row_sizes:
            col_offset = 0
            for c_size in col_sizes:
                block = results[idx]
                C[row_offset:row_offset+r_size, col_offset:col_offset+c_size] = block
                col_offset += c_size
                idx += 1
            row_offset += r_size
        return C
    else:
        raise ValueError(f"Unknown partition strategy: {partition}")

