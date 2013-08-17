import sys
import time
import numpy as np
from gemm_pyopencl import gemm_pyopencl_cpu


def main():
    Aorder, Border, M, N, K, dtype = sys.argv[1:]
    M, N, K = map(int, [M, N, K])
    alpha = 1.0
    beta = 0.0

    rng = np.random.RandomState(0)
    A = np.asarray(rng.normal(size=(M, K)), dtype=dtype, order=Aorder)
    B = np.asarray(rng.normal(size=(K, N)), dtype=dtype, order=Border)
    C = np.asarray(rng.normal(size=(M, N)), dtype=dtype)
    C2 = np.empty_like(C)
    FLOPS = M * N * (K + 1) * 2
    times = []
    for i in range(10):
        rng.seed(i)
        C[:] = np.asarray(rng.normal(size=(M, N)), dtype=dtype)

        t0 = time.time()
        gemm_pyopencl_cpu(alpha, A, B, beta, C)
        t1 = time.time()
        print 'pyopencl time: ', (t1 - t0), 'GFlops', FLOPS / (t1 - t0) / (1000 ** 3)
        times.append(t1 - t0)

        if 0:
            rng.seed(i)
            C2[:] = np.asarray(rng.normal(size=(M, N)), dtype=dtype)

            t0 = time.time()
            C3 = alpha * np.dot(A, B) + beta * C2
            t1 = time.time()
            print 'np.dot time:   ', (t1 - t0), 'GFlops', FLOPS / (t1 - t0) / (1000 ** 3)

            if not np.allclose(C, C3, atol=1e-3, rtol=1e-3):
                print np.max(abs(C - C3))
                raise ValueError('Computed wrong answer')

    print M, N, K, dtype,
    print 'best GFLOP/s', FLOPS / min(times) / (1000 ** 3)
    print M, N, K, dtype,
    print 'avg  GFLOP/s', FLOPS / np.mean(times) / (1000 ** 3)


if __name__ == '__main__':
    sys.exit(main())

