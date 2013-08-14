import time
import numpy as np
from numpy.linalg import norm
import sys
import reikna.cluda as cluda
from reikna.matrixmul import MatrixMul

api = cluda.ocl_api()
amd, = api.get_platforms()
gpu_dev, cpu_dev = amd.get_devices()
thr = api.Thread(cpu_dev)
print 'USING DEVICE: ', thr._device


def main(M=512, N=512, K=512, seed=0, dtype=np.float32):
    rng = np.random.RandomState(seed)
    A = np.asarray(rng.normal(size=(M, K)), dtype=dtype)
    B = np.asarray(rng.normal(size=(K, N)), dtype=dtype)
    C = np.asarray(rng.normal(size=(M, N)), dtype=dtype)
    C_reference = np.dot(A, B)
    a_dev = thr.to_device(A)
    b_dev = thr.to_device(B)
    c_dev = thr.to_device(C)

    dot = MatrixMul(a_dev, b_dev, out_arr=c_dev)
    dotc = dot.compile(thr)

    FLOPS = M * N * (K + 1) * 2

    for i in range(5):
        C[:] = 0

        t0 = time.time()
        dotc(c_dev, a_dev, b_dev)
        c_val = c_dev.get()
        t1 = time.time()
        print 'reikna time: ', (t1 - t0), 'GFlops', FLOPS / (t1 - t0) / (1000 ** 3)
        assert np.allclose(c_val, C_reference)

if __name__ == '__main__':
    sys.exit(main())
