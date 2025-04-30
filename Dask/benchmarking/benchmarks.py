import time
import numpy as np
from project import utils

def run_matrix_benchmark(max_gpus=None, M=2048, N=2048, K=2048, partition='row'):
    """
    Benchmark matrix multiplication on 1 up to max_gpus GPUs. Generates random matrices of size MxK and KxN,
    times the multiplication, and prints/saves speedups.
    """
    if max_gpus is None:
        max_gpus = utils.available_gpus()
        if max_gpus < 1:
            raise RuntimeError("No GPU available for benchmarking.")
    max_gpus = max(1, max_gpus)
    # Prepare input matrices
    A = np.random.rand(M, K).astype(np.float32)
    B = np.random.rand(K, N).astype(np.float32)
    results = []
    # Baseline: 1 GPU
    t0 = time.perf_counter()
    C_base = utils.run_matrix_multiply_kernel(A, B)
    # Ensure result is materialized on host for fair timing
    C_base_host = np.array(C_base) if isinstance(C_base, np.ndarray) else utils.cp.asnumpy(C_base)
    t1 = time.perf_counter()
    base_time = t1 - t0
    results.append((1, 1.0, base_time))
    print(f"Baseline (1 GPU) time: {base_time:.2f} s")
    # Determine GPU counts to test (powers of 2 and max_gpus)
    gpu_counts = []
    p = 1
    while p < max_gpus:
        p *= 2
        gpu_counts.append(p)
    if max_gpus not in gpu_counts:
        gpu_counts.append(max_gpus)
    gpu_counts = sorted({g for g in gpu_counts if g <= max_gpus and g > 1})
    # Run multi-GPU tests
    for g in gpu_counts:
        cluster, client = utils.start_cluster(g)
        start = time.perf_counter()
        C = utils.execute_distributed_multiply(A, B, client, partition=partition)
        client.close()
        cluster.close()
        end = time.perf_counter()
        elapsed = end - start
        speedup = base_time / elapsed
        results.append((g, speedup, elapsed))
        print(f"{g} GPUs time: {elapsed:.2f} s, speedup: {speedup:.2f}x")
    # Save results to CSV
    lines = ["GPUs,Speedup,Time(sec)"]
    for (g, s, t) in results:
        lines.append(f"{g},{s:.2f},{t:.2f}")
    with open("benchmarking/benchmark_results.csv", "w") as f:
        f.write("\n".join(lines))
    print("Benchmark results saved to benchmarking/benchmark_results.csv")
    return results

