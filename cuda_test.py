from numba import cuda 
import numpy as np 

@cuda.jit
def add_kernel(a, b, c , rows , cols):
    i,j = cuda.grid(2)
    if i < rows and j < cols:
        c[i, j] = a[i, j] + b[i, j]


def gpu_matrix_add(a , b ):

    if a.shape != b.shape:
        raise ValueError("Input matrices must have the same shape")
    
    if a.datatype != np.float32:
        a = a.astype(np.float32)
    
    if b.datatype != np.float32:
        b = b.astype(np.float32)

    rows, cols = a.shape

    threadsperblock = (16, 16)
    blockspergrid_x = (rows + (threadsperblock[0] - 1)) // threadsperblock[0]
    blockspergrid_y = (cols + (threadsperblock[1] - 1)) // threadsperblock[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    d_a = cuda.to_device(a)
    d_b = cuda.to_device(b)
    d_c = cuda.device_array((rows, cols), dtype=np.float32)


    import time 
    t0 = time.perf_counter()

    add_kernel[blockspergrid, threadsperblock](d_a, d_b, d_c , rows , cols)
    cuda.synchronize()

    c=d_c.copy_to_host()

    t1 = time.perf_counter()
    elapsed = t1 - t0
    print(f"Time taken for GPU addition: {elapsed} seconds")

    return c , elapsed 