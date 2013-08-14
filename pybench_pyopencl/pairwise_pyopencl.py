import sys
import time
import numpy as np
from mako.template import Template
import pyopencl as cl
from gemm_pyopencl import memoize, BlockingError, StrideError
from gemm_pyopencl import elemstrides, ctype_from_dtype
mf = cl.mem_flags
PROFILING = 0

ctx = cl.create_some_context()
if PROFILING:
    queue = cl.CommandQueue(
        ctx,
        properties=cl.command_queue_properties.PROFILING_ENABLE)
else:
    queue = cl.CommandQueue(ctx)


# TODO: figure out how to patch in the difference
#       calculation into the GEMM code generator

vectorized_text_CF =  """
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void kern(__global ${ctype}${KB} *A,
                  __global ${ctype}${KB} *B,
                  __global ${ctype}${NB} *C)
{
  const ${ctype}${NB} beta = (${ctype}${NB})(
      ${beta}
  % for ii in range(NB - 1):
      , ${beta}
  % endfor
  );
  % for mi in range(MB):
  ${ctype}${KB} Abuf${mi};
  % endfor

  ${ctype}${KB} Bbuf, diff;

  % for mi in range(MB):
      ${ctype}${NB} Cbuf${mi};
  % endfor

  for(int mb = get_global_id(0); mb < ${NoMB}; mb += get_global_size(0))
  {
    for(int nb = get_global_id(1); nb < ${NoNB}; nb += get_global_size(1))
    {
      % for mi in range(MB):
      Cbuf${mi} = (${ctype}${NB})(
                    0
                    % for foo in range(NB - 1):
                    , 0
                    % endfor
                    );
      % endfor

      for (int kb = 0; kb < ${NoKB}; ++kb)
      {
        // load MB K-blocks of A
        % for mi in range(MB):
            Abuf${mi} = A[${As0} * (mb * ${MB} + ${mi}) + kb];
        % endfor

        // load NB K-blocks of B
        % for ni in range(NB):
            Bbuf = B[kb + ${Bs1} * (nb * ${NB} + ${ni})];

            % for mi in range(MB):
                diff = Bbuf - Abuf${mi};
                Cbuf${mi}.s${'%x' % ni} += dot(diff, diff);
            % endfor
        % endfor
      }

      % for mi in range(MB):
          C[(mb * ${MB} + ${mi}) * ${Cs0} + nb] = sqrt(Cbuf${mi});
      % endfor
    }
  }
}
    """

@memoize
def pairwise_cpu_prepare_vectorized_CF(alpha, beta, M, N, K, dtype,
                               Astrides, Bstrides, Cstrides, MB, NB, KB):
    ctype = ctype_from_dtype(dtype)
    (As0, As1) = elemstrides(Astrides, dtype, KB, vecdim=1)
    (Bs0, Bs1) = elemstrides(Bstrides, dtype, KB, vecdim=0)
    (Cs0, Cs1) = elemstrides(Cstrides, dtype, NB, vecdim=1)
    NoMB = M // MB
    NoNB = N // NB
    NoKB = K // KB
    if M != MB * NoMB:
        raise BlockingError()
    if N != NB * NoNB:
        raise BlockingError()
    if K != KB * NoKB:
        raise BlockingError()
    if NB > 16:
        raise BlockingError('codegen breaks at this point')
    text = Template(vectorized_text_CF, output_encoding='ascii').render(**locals())
    if 0:
        for ii, line in enumerate(text.split('\n')):
            print ii, line
    prg = cl.Program(ctx, text).build()
    #print 'built!'
    return prg.kern


comptimes = []
def pairwise_pyopencl_cpu(A, B, C):
    kern = None
    alpha = 1
    beta = 0

    # TODO: patch GEMM code generators

    # TODO: predictive auto-tuning here, not search
    #       answer should depend on dtype
    for MB in [4, 2]:
        for NB in [2]:
            for KB in [4, 2]:
                if kern:
                    continue
                try:
                    kern = pairwise_cpu_prepare_vectorized_CF(
                        alpha, beta,
                        C.shape[0], C.shape[1], A.shape[1],
                        A.dtype,
                        A.strides, B.strides, C.strides,
                        MB=MB, NB=NB, KB=KB)
                    global_shape = (8, 8)   # enough for different cores
                    local_shape = (1, 1)
                    #print 'Using kernel for MB=%i NB=%i KB=%i' % (MB, NB, KB)
                except StrideError:
                    pass

    A_buf = cl.Buffer(ctx, mf.USE_HOST_PTR, hostbuf=A)
    B_buf = cl.Buffer(ctx, mf.USE_HOST_PTR, hostbuf=B)
    C_buf = cl.Buffer(ctx, mf.USE_HOST_PTR, hostbuf=C)
    ev = kern(queue, global_shape, local_shape, A_buf, B_buf, C_buf)
    #cl.enqueue_copy(queue, C, C_buf)
    queue.finish()
    if PROFILING:
        comptimes.append(1e-9 * (ev.profile.end - ev.profile.start))
        print 'computation time', min(comptimes)


def main(M=300, N=300, K=150, seed=0, dtype=np.float64):
    rng = np.random.RandomState(seed)
    A = np.asarray(rng.normal(size=(M, K)), dtype=dtype)
    B = np.asarray(rng.normal(size=(K, N)), dtype=dtype, order='F')
    C = np.asarray(rng.normal(size=(M, N)), dtype=dtype)
    ref = np.sqrt(((A[:, None, :] - B.T) ** 2).sum(axis=2))
    FLOPS = M * N * (K + 1) * 3
    times = []
    for i in range(50):
        C[:] = 0

        t0 = time.time()
        pairwise_pyopencl_cpu(A, B, C)
        t1 = time.time()
        print M, N, K, dtype, 'pyopencl time: ',
        print (t1 - t0), 'GFlops', FLOPS / (t1 - t0) / (1000 ** 3)
        times.append(t1 - t0)

        if not np.allclose(ref, C):
            print np.max(abs(ref - C))
            raise ValueError('wrong answer', ref, C)


    print M, N, K, dtype,
    print 'best GFLOP/s', FLOPS / min(times) / (1000 ** 3)
    print M, N, K, dtype,
    print 'avg  GFLOP/s', FLOPS / np.mean(times) / (1000 ** 3)

if __name__ == '__main__':
    sys.exit(main())
