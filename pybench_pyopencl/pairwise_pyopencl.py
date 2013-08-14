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


vectorized_text =  """
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void kern(__global ${ctype}${KB} *A,
                  __global ${ctype}${NB} *B,
                  __global ${ctype}${NB} *C)
{
  ${ctype}${NB} tmp;
  % for ii in range(MB):
  ${ctype}${KB} Abuf${ii};
  ${ctype}${NB} Cbuf${ii};
  % endfor
  ${ctype}${NB} Bbuf;
  for(int mb = get_global_id(0); mb < ${NoMB}; mb += get_global_size(0))
  {
    for(int nb = get_global_id(1); nb < ${NoNB}; nb += get_global_size(1))
    {
      % for ii in range(MB):
      Cbuf${ii} = (${ctype}${NB})(
                    0
                    % for foo in range(NB - 1):
                    , 0
                    % endfor
                    );
      % endfor

      for (int kb = 0; kb < ${NoKB}; ++kb)
      {
        // load KB columns of A at a time
        % for ii in range(MB):
        Abuf${ii} = A[${As0} * (mb * ${MB} + ${ii}) + kb];
        % endfor

        % for kki in range(KB):
        Bbuf = B[(kb * ${KB} + ${kki}) * ${Bs0} + nb];

            % for ii in range(MB):

            tmp = (${ctype}${NB})(
                    Abuf${ii}.s${kki}
                    % for foo in range(NB - 1):
                    , Abuf${ii}.s${kki}
                    % endfor
                    );
            tmp -= Bbuf;
            Cbuf${ii} = mad(tmp, tmp, Cbuf${ii});
            % endfor

        % endfor
      }

      % for ii in range(MB):
          C[(mb * ${MB} + ${ii}) * ${Cs0} + nb] = sqrt(Cbuf${ii});
      % endfor
    }
  }
}
    """

@memoize
def pairwise_cpu_prepare_vectorized(M, N, K, dtype,
                                    Astrides, Bstrides, Cstrides,
                                    MB, NB, KB):
    ctype = ctype_from_dtype(dtype)
    (As0, As1) = elemstrides(Astrides, dtype, KB, vecdim=1)
    (Bs0, Bs1) = elemstrides(Bstrides, dtype, NB, vecdim=1)
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
    text = Template(vectorized_text, output_encoding='ascii').render(**locals())
    if 0:
        for ii, line in enumerate(text.split('\n')):
            print ii, line
    prg = cl.Program(ctx, text).build()
    if 0:
        print 'built!'

    return prg.kern


comptimes = []
def pairwise_pyopencl_cpu(A, B, C):
    A = np.asarray(A, order='C')
    B = np.asarray(B, order='C')
    if C.strides[1] not in (4, 8):
        raise NotImplementedError('output array not row-major')
    kern = None
    global_shape = (4, 4)   # enough for different cores
    local_shape = (1, 1)    # I think this does nothing on CPU (true?)

    # TODO: cache the result of the search
    for MB in [16, 8, 4, 2]:
        for NB in [16, 8, 4, 2]:
            for KB in [16, 8, 4, 2]:
                if kern:
                    continue
                try:
                    kern = pairwise_cpu_prepare_vectorized(
                        C.shape[0], C.shape[1], A.shape[1],
                        A.dtype,
                        A.strides, B.strides, C.strides,
                        MB=MB, NB=NB, KB=KB)
                    #print 'Using kernel for MB=%i NB=%i KB=%i' % (MB, NB, KB)
                except (StrideError, BlockingError):
                    pass

    A_buf = cl.Buffer(ctx, mf.COPY_HOST_PTR, hostbuf=A)
    B_buf = cl.Buffer(ctx, mf.COPY_HOST_PTR, hostbuf=B)
    C_buf = cl.Buffer(ctx, mf.COPY_HOST_PTR, hostbuf=C) # --copy not necessary
    ev = kern(queue, global_shape, local_shape, A_buf, B_buf, C_buf)
    cl.enqueue_copy(queue, C, C_buf)
    queue.finish()
    if PROFILING:
        comptimes.append(1e-9 * (ev.profile.end - ev.profile.start))
        print 'computation time', min(comptimes)


def main(M=300, N=300, K=150, seed=0, dtype=np.float64):
    rng = np.random.RandomState(seed)
    A = np.asarray(rng.normal(size=(M, K)), dtype=dtype)
    B = np.asarray(rng.normal(size=(K, N)), dtype=dtype)
    C = np.asarray(rng.normal(size=(M, N)), dtype=dtype)
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


    print M, N, K, dtype,
    print 'best GFLOP/s', FLOPS / min(times) / (1000 ** 3)
    print M, N, K, dtype,
    print 'avg  GFLOP/s', FLOPS / np.mean(times) / (1000 ** 3)

if __name__ == '__main__':
    sys.exit(main())
