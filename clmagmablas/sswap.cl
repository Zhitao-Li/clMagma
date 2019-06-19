/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date
       
       @author Mark Gates

       @generated from clmagmablas/zswap.cl, normal z -> s, Tue Jun 18 16:14:14 2019

       auto-converted from sswap.cu

*/
#include "kernels_header.h"
#include "sswap.h"


/* Vector is divided into ceil(n/nb) blocks.
   Each thread swaps one element, x[tid] <---> y[tid].
*/
__kernel void sswap_kernel(
    magma_int_t n,
    __global float *x, unsigned long x_offset, magma_int_t incx,
    __global float *y, unsigned long y_offset, magma_int_t incy )
{
    x += x_offset;
    y += y_offset;

    float tmp;
    int ind = get_local_id(0) + get_local_size(0)*get_group_id(0);
    if ( ind < n ) {
        x += ind*incx;
        y += ind*incy;
        tmp = *x;
        *x  = *y;
        *y  = tmp;
    }
}
