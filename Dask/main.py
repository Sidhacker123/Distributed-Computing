import argparse
import numpy as np
from Dask import utils
from Dask.benchmarking import benchmarks

def main():
    parser = argparse.ArgumentParser(description="GPU Matrix Computation Project")
    parser.add_argument("--m", type=int, help="Number of rows of matrix A (and C)", default=1024)
    parser.add_argument("--n", type=int, help="Number of columns of matrix B (and C)", default=1024)
    parser.add_argument("--k", type=int, help="Number of columns of A / rows of B", default=1024)
    parser.add_argument("--gpus", type=int, help="Number of GPUs to use (default: 1, or all available if --benchmark)", default=None)
    parser.add_argument("--partition", choices=["row", "block"], default="row", help="Partitioning strategy for multi-GPU (default: row)")
    parser.add_argument("--benchmark", action="store_true", help="Run scalability benchmark")
    parser.add_argument("--profile", action="store_true", help="Profile execution time for a single run")
    args = parser.parse_args()

    # Determine number of GPUs to use
    num_gpus = args.gpus if args.gpus is not None else (utils.available_gpus() if args.benchmark else 1)
    if num_gpus < 1:
        raise RuntimeError("No GPUs available.")
    M, N, K = args.m, args.n, args.k
    partition = args.partition

    if args.benchmark:
        # Run benchmark suite across 1..N GPUs
        print(f"Running benchmark up to {num_gpus} GPUs (Matrix size: {M}x{K} * {K}x{N}, partition: {partition})")
        benchmarks.run_matrix_benchmark(max_gpus=num_gpus, M=M, N=N, K=K, partition=partition)
    else:
        # Single-run execution (can be multi-GPU or single GPU)
        print(f"Executing matrix multiplication: A({M}x{K}) * B({K}x{N}) using {num_gpus} GPU(s) [partition={partition}]")
        # Generate input matrices
        A = np.random.rand(M, K).astype(np.float32)
        B = np.random.rand(K, N).astype(np.float32)
        if num_gpus == 1:
            # Single GPU execution
            if args.profile:
                # Profile kernel and total time using CUDA events
                start_evt = utils.cp.cuda.Event() if hasattr(utils, 'cp') else None
                end_evt = utils.cp.cuda.Event() if hasattr(utils, 'cp') else None
                if start_evt and end_evt:
                    start_evt.record()
                # Run matrix multiplication on GPU (with internal kernel timing)
                C_gpu, kernel_ms = utils.run_matrix_multiply_kernel(A, B, measure_time=True)
                if start_evt and end_evt:
                    end_evt.record()
                    end_evt.synchronize()
                    total_time_ms = utils.cp.cuda.get_elapsed_time(start_evt, end_evt)
                else:
                    total_time_ms = None
                # Copy result back to host and report times
                C = utils.cp.asnumpy(C_gpu)
                if total_time_ms is not None:
                    print(f"Total time: {total_time_ms/1000:.4f} s (Kernel time: {kernel_ms:.4f} ms)")
                else:
                    print(f"Kernel time: {kernel_ms:.4f} ms")
            else:
                # Execute without profiling
                C_gpu = utils.run_matrix_multiply_kernel(A, B)
                C = utils.cp.asnumpy(C_gpu)
                print(f"Result matrix shape: {C_gpu.shape}")
        else:
            # Multi-GPU execution using Dask
            cluster, client = utils.start_cluster(num_gpus)
            import time
            t_start = time.perf_counter() if args.profile else None
            C = utils.execute_distributed_multiply(A, B, client, partition=partition)
            if args.profile and t_start is not None:
                t_end = time.perf_counter()
                print(f"Total distributed time: {t_end - t_start:.4f} s")
            else:
                print(f"Result matrix shape: {C.shape}")
            client.close()
            cluster.close()

if __name__ == "__main__":
    main()

