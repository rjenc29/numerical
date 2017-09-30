from numba import cuda
import numba
from numba import float32


@numba.cuda.jit("void(float32[:,:], float32[:,:], float32[:,:])")
def naive_matrix_mult(A, B, C):

    n = A.shape[0]

    x, y = cuda.grid(2)
    if x >= n or y >= n:
        return

    C[y, x] = 0
    for i in range(n):
        C[y, x] += A[y, i] * B[i, x]


@numba.cuda.jit("void(float32[:,:], float32[:,:], float32[:,:])")
def optimised_matrix_mult(A, B, C):

    n = A.shape[0]
    threads_per_block = 32
    shared_mem_size = (threads_per_block, threads_per_block)
    blocks_per_grid = int(n / threads_per_block)

    # Declare shared memory
    sA = cuda.shared.array(shape=shared_mem_size, dtype=float32)
    sB = cuda.shared.array(shape=shared_mem_size, dtype=float32)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    x, y = cuda.grid(2)

    acc = 0
    for i in range(blocks_per_grid):
        if x < n and y < n:
            # Prefill cache
            sA[ty, tx] = A[y, tx + i * threads_per_block]
            sB[ty, tx] = B[ty + i * threads_per_block, x]

        # Synchronize all threads in the block
        cuda.syncthreads()

        if x < n and y < n:
            # Compute product
            for j in range(threads_per_block):
                acc += sA[ty, j] * sB[j, tx]

        # Wait until all threads finish the computation
        cuda.syncthreads()

    if x < n and y < n:
        C[y, x] = acc


if __name__ == '__main__':
    import pytest
    pytest.main()



