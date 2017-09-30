import utilities.cuda_matrix_multiplication as kernels
import numpy as np
from numba import cuda
import numba
from timeit import default_timer as timer


def _test_kernel(kernel):

    # please excuse the print statement - pycharm / pytest config issues...

    device = cuda.get_current_device()
    # print(device)

    n = 32 * 32 * 3  # any bigger and kernel crashes

    # Prepare data on the CPU
    A = np.array(np.random.random((n, n)), dtype=np.float32)
    B = np.array(np.random.random((n, n)), dtype=np.float32)

    # Prepare data on the GPU
    dA = cuda.to_device(A)
    dB = cuda.to_device(B)
    dC = cuda.device_array_like(A)

    # threading and blocking strategy
    threads_per_block = device.WARP_SIZE
    blocks_per_grid = int(n / threads_per_block)

    griddim = blocks_per_grid, blocks_per_grid
    blockdim = threads_per_block, threads_per_block

    # time execution
    s = timer()
    kernel[griddim, blockdim](dA, dB, dC)
    numba.cuda.synchronize()
    e = timer()
    output = dC.copy_to_host()
    elapsed_cuda = e - s

    s = timer()
    expected = A @ B
    e = timer()
    elapsed_numpy = e - s

    assert np.allclose(expected, output)
    # print('Matrix of two ({0} x {0}) matrices:\nnumpy: {1} | cuda: {2}'.format(n, elapsed_numpy, elapsed_cuda))


def test_naive_kernel():
    _test_kernel(kernels.naive_matrix_mult)
    # slightly slower than raw numpy

def test_optimised_kernel():
    _test_kernel(kernels.optimised_matrix_mult)
    # substantially slower than naive


if __name__ == '__main__':
    import pytest
    pytest.main()
