/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @generated from clmagmablas/zcaxpycp.cl, mixed zc -> ds, Tue Jun 18 16:14:14 2019

       auto-converted from dsaxpycp.cu

*/
#include "kernels_header.h"
#include "dsaxpycp.h"

// adds   x += r (including conversion to double)  --and--
// copies w = b
// each thread does one index, x[i] and w[i]
__kernel void
dsaxpycp_kernel(
    magma_int_t m,
    __global float *r, unsigned long r_offset,
    __global double *x, unsigned long x_offset,
    __global const double *b, unsigned long b_offset,
    __global double *w, unsigned long w_offset )
{
    r += r_offset;
    x += x_offset;
    b += b_offset;
    w += w_offset;

    const int i = get_local_id(0) + get_group_id(0)*NB;
    if ( i < m ) {
        x[i] = MAGMA_D_ADD( x[i], MAGMA_D_MAKE( MAGMA_D_REAL( r[i] ),
                                                MAGMA_D_IMAG( r[i] ) ) );
        w[i] = b[i];
    }
}
