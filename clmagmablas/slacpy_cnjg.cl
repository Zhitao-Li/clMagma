/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @generated from clmagmablas/zlacpy_cnjg.cl, normal z -> s, Tue Jun 18 16:14:14 2019

       auto-converted from slacpy_cnjg.cu

*/
#include "kernels_header.h"
#include "slacpy_cnjg.h"

// copy & conjugate a single vector of length n.
// TODO: this was modeled on the old sswap routine. Update to new slacpy code for 2D matrix?

__kernel void slacpy_cnjg_kernel(
    magma_int_t n,
    __global float *A1, unsigned long A1_offset, magma_int_t lda1,
    __global float *A2, unsigned long A2_offset, magma_int_t lda2 )
{
    A1 += A1_offset;
    A2 += A2_offset;

    int x = get_local_id(0) + get_local_size(0)*get_group_id(0);
    int offset1 = x*lda1;
    int offset2 = x*lda2;
    if ( x < n )
    {
        A2[offset2] = MAGMA_S_CNJG( A1[offset1] );
    }
}
