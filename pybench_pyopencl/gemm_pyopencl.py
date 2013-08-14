# Authors: James Bergstra
# License: BSD-3

import sys
import time

from mako.template import Template
import numpy as np
import pyopencl as cl

mf = cl.mem_flags

PROFILING = 0

ctx = cl.create_some_context()
if PROFILING:
    queue = cl.CommandQueue(
        ctx,
        properties=cl.command_queue_properties.PROFILING_ENABLE)
else:
    queue = cl.CommandQueue(ctx)


def ctype_from_dtype(dtype):
    return {
            'float32': 'float',
            'float64': 'double',
            }[str(dtype)]


def elemstrides(strides, dtype, vec=1, vecdim=-1):
    size = {
        'float32': 4,
        'float64': 8,
    }[str(dtype)]
    # TODO: raise StrideError
    for stride in strides:
        assert stride % size == 0
    val_strides = tuple(int(s / size) for s in strides)
    if vec == 1:
        return val_strides
    vecdim = range(len(strides))[vecdim]   # -- make vecdim non-neg
    for ii, val_stride in enumerate(val_strides):
        if ii == vecdim:
            if val_stride != 1:
                raise StrideError()
        else:
            if val_stride % vec:
                raise StrideError()
    vec_strides = [int(s / vec) for s in val_strides]
    vec_strides[vecdim] = 1
    return tuple(vec_strides)



def memoize(f):
    cache = {}
    def new_fn(*args, **kwargs):
        key = args + tuple(sorted(kwargs.items()))
        try:
            return cache[key]
        except KeyError:
            rval = f(*args, **kwargs)
            cache[key] = rval
            return rval
    new_fn.__name__ = f.__name__
    new_fn.memoize_cache = cache
    return new_fn


@memoize
def gemm_cpu_prepare_reference(alpha, beta, M, N, K, dtype,
                               Astrides, Bstrides, Cstrides):
    ctype = ctype_from_dtype(dtype)
    (As0, As1) = elemstrides(Astrides, dtype)
    (Bs0, Bs1) = elemstrides(Bstrides, dtype)
    (Cs0, Cs1) = elemstrides(Cstrides, dtype)
    prg = cl.Program(ctx, """
        __kernel void ref(__global %(ctype)s *A,
                          __global %(ctype)s *B,
                          __global %(ctype)s *C)
        {
          for(int mm = get_global_id(0); mm < %(M)s; mm += get_global_size(0))
          {
            for(int nn = get_global_id(1); nn < %(N)s; nn += get_global_size(1))
            {
              %(ctype)s buf = 0;
              for (int kk = 0; kk < %(K)s; ++kk)
              {
                  buf += A[mm * %(As0)s + kk * %(As1)s]
                       * B[kk * %(Bs0)s + nn * %(Bs1)s];
              }
              C[mm * %(Cs0)s + nn * %(Cs1)s] *= %(beta)s;
              C[mm * %(Cs0)s + nn * %(Cs1)s] += %(alpha)s * buf;
            }
          }
        }
        """ % locals()).build()

    return prg.ref

class StrideError(Exception):
    """StrideError"""

class BlockingError(Exception):
    """BlockingError"""

vectorized_text_CC =  """
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void kern(__global ${ctype}${KB} *A,
                  __global ${ctype}${NB} *B,
                  __global ${ctype}${NB} *C)
{
  const ${ctype}${NB} beta = (${ctype}${NB})(
      ${beta}
  % for ii in range(NB - 1):
      , ${beta}
  % endfor
  );
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
                    Abuf${ii}.s${'%x' % kki}
                    % for foo in range(NB - 1):
                    , Abuf${ii}.s${'%x' % kki}
                    % endfor
                    );
            Cbuf${ii} = mad(tmp, Bbuf, Cbuf${ii});
            % endfor

        % endfor
      }

      % if alpha != 1:
          % for ii in range(MB):
              Cbuf${ii} *= (${ctype}${NB})(
                    ${alpha}
                    % for foo in range(NB - 1):
                    , ${alpha}
                    % endfor
                    );
          % endfor
      % endif

      % for ii in range(MB):
          % if beta == 0:
              C[(mb * ${MB} + ${ii}) * ${Cs0} + nb] = Cbuf${ii};
          % elif beta == 1:
              C[(mb * ${MB} + ${ii}) * ${Cs0} + nb] += Cbuf${ii};
          % else:
              C[(mb * ${MB} + ${ii}) * ${Cs0} + nb] = mad(beta, C[(mb* ${MB} + ${ii}) * ${Cs0} + nb], Cbuf${ii});
          % endif
      % endfor
    }
  }
}
    """


@memoize
def gemm_cpu_prepare_vectorized_CC(alpha, beta, M, N, K, dtype,
                               Astrides, Bstrides, Cstrides, MB, NB, KB):
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
    if KB > 16:
        raise BlockingError('codegen breaks at this point')
    text = Template(vectorized_text_CC, output_encoding='ascii').render(**locals())
    if 0:
        for ii, line in enumerate(text.split('\n')):
            print ii, line
    prg = cl.Program(ctx, text).build()
    #print 'built!'

    return prg.kern


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

  ${ctype}${KB} Bbuf;

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
                Cbuf${mi}.s${'%x' % ni} += dot(Bbuf, Abuf${mi});
            % endfor
        % endfor
      }

      % if alpha != 1:
          % for ii in range(MB):
              Cbuf${ii} *= (${ctype}${NB})(
                    ${alpha}
                    % for foo in range(NB - 1):
                    , ${alpha}
                    % endfor
                    );
          % endfor
      % endif

      % for ii in range(MB):
          % if beta == 0:
              C[(mb * ${MB} + ${ii}) * ${Cs0} + nb] = Cbuf${ii};
          % elif beta == 1:
              C[(mb * ${MB} + ${ii}) * ${Cs0} + nb] += Cbuf${ii};
          % else:
              C[(mb * ${MB} + ${ii}) * ${Cs0} + nb] = mad(beta, C[(mb* ${MB} + ${ii}) * ${Cs0} + nb], Cbuf${ii});
          % endif
      % endfor
    }
  }
}
    """

@memoize
def gemm_cpu_prepare_vectorized_CF(alpha, beta, M, N, K, dtype,
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
def gemm_pyopencl_cpu(alpha, A, B, beta, C):
    kern = None
    # TODO: fix this to number of CPU cores, as read from cl

    # TODO: predictive auto-tuning here, not search
    #       answer should depend on dtype
    for MB in [32, 16, 8, 4, 2]:
        for NB in [8, 4, 2]:
            for KB in [4, 2]:
                if kern:
                    continue
                try:
                    kern = gemm_cpu_prepare_vectorized_CC(
                        alpha, beta,
                        C.shape[0], C.shape[1], A.shape[1],
                        A.dtype,
                        A.strides, B.strides, C.strides,
                        MB=MB, NB=NB, KB=KB)
                    global_shape = (8, 1)   # enough for different cores
                    local_shape = (1, 1)
                    print 'Using kernel for MB=%i NB=%i KB=%i' % (MB, NB, KB)
                except StrideError:
                    pass
    for MB in [4, 2]:
        for NB in [2]:
            for KB in [4, 2]:
                if kern:
                    continue
                try:
                    kern = gemm_cpu_prepare_vectorized_CF(
                        alpha, beta,
                        C.shape[0], C.shape[1], A.shape[1],
                        A.dtype,
                        A.strides, B.strides, C.strides,
                        MB=MB, NB=NB, KB=KB)
                    global_shape = (8, 8)   # enough for different cores
                    local_shape = (1, 1)
                    print 'Using kernel for MB=%i NB=%i KB=%i' % (MB, NB, KB)
                except StrideError:
                    pass
    if kern is None:
        print 'Using reference kernel'
        kern = gemm_cpu_prepare_reference(alpha, beta,
                                          C.shape[0], C.shape[1], A.shape[1],
                                          A.dtype,
                                          A.strides, B.strides, C.strides)
        global_shape = (8, 8)   # enough for different cores
        local_shape = (1, 1)

    A_buf = cl.Buffer(ctx, mf.COPY_HOST_PTR, hostbuf=A)
    B_buf = cl.Buffer(ctx, mf.COPY_HOST_PTR, hostbuf=B)
    C_buf = cl.Buffer(ctx, mf.COPY_HOST_PTR, hostbuf=C)
    ev = kern(queue, global_shape, local_shape, A_buf, B_buf, C_buf)
    cl.enqueue_copy(queue, C, C_buf)
    queue.finish()
    if PROFILING:
        comptimes.append(1e-9 * (ev.profile.end - ev.profile.start))
        print 'computation time', min(comptimes)


benchmarks = (
    gemm_pyopencl_cpu,
)

def mainCC1024(M=1024, N=1024, K=1024, seed=0, dtype=np.float64):
    rng = np.random.RandomState(seed)
    A = np.asarray(rng.normal(size=(M, K)), dtype=dtype)
    B = np.asarray(rng.normal(size=(K, N)), dtype=dtype)
    C = np.asarray(rng.normal(size=(M, N)), dtype=dtype)
    C2 = np.empty_like(C)
    Z = np.ones(10 * 1024 * 1024)
    alpha = 1.0
    beta = 0.0
    FLOPS = M * N * (K + 1) * 2
    times = []
    for i in range(20):
        rng.seed(i)
        C[:] = np.asarray(rng.normal(size=(M, N)), dtype=dtype)

        t0 = time.time()
        gemm_pyopencl_cpu(alpha, A, B, beta, C)
        t1 = time.time()
        print 'pyopencl time: ', (t1 - t0), 'GFlops', FLOPS / (t1 - t0) / (1000 ** 3)
        times.append(t1 - t0)
        continue

        rng.seed(i)
        C2[:] = np.asarray(rng.normal(size=(M, N)), dtype=dtype)

        t0 = time.time()
        C3 = alpha * np.dot( A, B) + beta * C2
        t1 = time.time()
        print 'np.dot time:   ', (t1 - t0), 'GFlops', FLOPS / (t1 - t0) / (1000 ** 3)

        print np.max(abs(C - C3))
        assert np.allclose(C, C3)
        # -- clear processor cache
        Z.sum()
        time.sleep(1)

    print M, N, K, dtype,
    print 'best GFLOP/s', FLOPS / min(times) / (1000 ** 3)
    print M, N, K, dtype,
    print 'avg  GFLOP/s', FLOPS / np.mean(times) / (1000 ** 3)

def mainCC2K():
    return mainCC1024(M=2048, N=2048, K=2048)

def mainCC512():
    return mainCC1024(M=512, N=512, K=512)

def mainCC256():
    return mainCC1024(M=256, N=256, K=256)

def mainCC128():
    return mainCC1024(M=128, N=128, K=128)

def mainCC2Kf():
    return mainCC1024(M=2048, N=2048, K=2048, dtype=np.float32)

def mainCC1024f():
    return mainCC1024(M=1024, N=1024, K=1024, dtype=np.float32)

def mainCC512f():
    return mainCC1024(M=512, N=512, K=512, dtype=np.float32)

def mainCC128f():
    return mainCC1024(M=128, N=128, K=128, dtype=np.float32)

def mainCF1024(M=1024, N=1024, K=1024, seed=0, dtype=np.float64,
        alpha=1.2,
        beta=0.1):
    rng = np.random.RandomState(seed)
    A = np.asarray(rng.normal(size=(M, K)), dtype=dtype)
    B = np.asarray(rng.normal(size=(K, N)), dtype=dtype, order='F')
    C = np.asarray(rng.normal(size=(M, N)), dtype=dtype)
    C2 = np.empty_like(C)
    FLOPS = M * N * (K + 1) * 2
    times = []
    for i in range(20):
        rng.seed(i)
        C[:] = np.asarray(rng.normal(size=(M, N)), dtype=dtype)

        t0 = time.time()
        gemm_pyopencl_cpu(alpha, A, B, beta, C)
        t1 = time.time()
        print 'pyopencl time: ', (t1 - t0), 'GFlops', FLOPS / (t1 - t0) / (1000 ** 3)
        times.append(t1 - t0)

        rng.seed(i)
        C2[:] = np.asarray(rng.normal(size=(M, N)), dtype=dtype)

        t0 = time.time()
        C3 = alpha * np.dot( A, B) + beta * C2
        t1 = time.time()
        print 'np.dot time:   ', (t1 - t0), 'GFlops', FLOPS / (t1 - t0) / (1000 ** 3)

        if not np.allclose(C, C3, atol=1e-3, rtol=1e-3):
            print np.max(abs(C - C3))
            raise ValueError('wrong answer', C, C3)
        # -- clear processor cache

    print M, N, K, dtype,
    print 'best GFLOP/s', FLOPS / min(times) / (1000 ** 3)
    print M, N, K, dtype,
    print 'avg  GFLOP/s', FLOPS / np.mean(times) / (1000 ** 3)

def mainCF512():
    return mainCF1024(M=512, N=512, K=512, dtype=np.float64)

def mainCF128():
    return mainCF1024(M=128, N=128, K=128, dtype=np.float64)

def mainCF1024f():
    return mainCF1024(M=1024, N=1024, K=1024, dtype=np.float32)

def mainCF512f():
    return mainCF1024(M=512, N=512, K=512, dtype=np.float32)

def mainCF128f():
    return mainCF1024(M=128, N=128, K=128, dtype=np.float32)

if __name__ == '__main__':
    sys.exit(globals()[sys.argv[1]]())
