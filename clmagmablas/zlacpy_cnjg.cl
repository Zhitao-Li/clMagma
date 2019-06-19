/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c

       auto-converted from zlacpy_cnjg.cu

*/
#include "kernels_header.h"
#include "zlacpy_cnjg.h"

// copy & conjugate a single vector of length n.
// TODO: this was modeled on the old zswap routine. Update to new zlacpy code for 2D matrix?

__kernel void zlacpy_cnjg_kernel(
    magma_int_t n,
    __global magmaDoubleComplex *A1, unsigned long A1_offset, magma_int_t lda1,
    __global magmaDoubleComplex *A2, unsigned long A2_offset, magma_int_t lda2 )
{
    A1 += A1_offset;
    A2 += A2_offset;

    int x = get_local_id(0) + get_local_size(0)*get_group_id(0);
    int offset1 = x*lda1;
    int offset2 = x*lda2;
    if ( x < n )
    {
        A2[offset2] = MAGMA_Z_CNJG( A1[offset1] );
    }
}
