/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @generated from clmagmablas/zaxpycp.cl, normal z -> s, Tue Jun 18 16:14:14 2019

       auto-converted from saxpycp.cu

*/
#include "kernels_header.h"
#include "saxpycp.h"

// adds   x += r  --and--
// copies r = b
// each thread does one index, x[i] and r[i]
__kernel void
saxpycp_kernel(
    magma_int_t m,
    __global float *r, unsigned long r_offset,
    __global float *x, unsigned long x_offset,
    __global const float *b, unsigned long b_offset)
{
    r += r_offset;
    x += x_offset;
    b += b_offset;

    const int i = get_local_id(0) + get_group_id(0)*NB;
    if ( i < m ) {
        x[i] = MAGMA_S_ADD( x[i], r[i] );
        r[i] = b[i];
    }
}
