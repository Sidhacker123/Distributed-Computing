# Partitioning Strategy

## 1. Row-wise Sharding
Each MPI node receives contiguous row blocks from the input matrix A.
- ✅ Simpler communication pattern
- ❌ Imbalanced load if rows ≠ divisible by number of nodes

## 2. Block-wise Sharding
Each MPI node gets a square block (submatrix) of A and B.
- ✅ Balanced workload
- ✅ Better cache and GPU memory utilization
- ❌ Requires more complex communication (especially for border data)

## Strategy Used
In this implementation, row-wise partitioning was used for simplicity and ring communication via MPI.
