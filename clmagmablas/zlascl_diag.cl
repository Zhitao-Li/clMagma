/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c

       auto-converted from zlascl_diag.cu
*/
#include "kernels_header.h"
#include "zlascl_diag.h"


// each thread block does one NB x n block row of A.
// each thread does one row, starting from left edge and moving right to diagonal.
__kernel void
zlascl_diag_lower(
    magma_int_t m, magma_int_t n,
    __global const magmaDoubleComplex* D, unsigned long D_offset, magma_int_t ldd,
    __global magmaDoubleComplex*       A, unsigned long A_offset, magma_int_t lda)
{
    D += D_offset;
    A += A_offset;

    int ind = get_group_id(0) * NB + get_local_id(0);

    A += ind;
    if (ind < m) {
        for (int j=0; j < n; j++ ) {
            A[j*lda] = MAGMA_Z_DIV( A[j*lda], D[j + j*ldd] );
        }
    }
}


// each thread block does one NB x n block row of A.
// each thread does one row, starting from right edge and moving left to diagonal.
__kernel void
zlascl_diag_upper(
    magma_int_t m, magma_int_t n,
    __global const magmaDoubleComplex* D, unsigned long D_offset, magma_int_t ldd,
    __global magmaDoubleComplex*       A, unsigned long A_offset, magma_int_t lda)
{
    D += D_offset;
    A += A_offset;

    int ind = get_group_id(0) * NB + get_local_id(0);

    A += ind;
    if (ind < m) {
        for (int j=0; j < n; j++ ) {
            A[j*lda] = MAGMA_Z_DIV( A[j*lda], D[ind + ind*ldd] );
        }
    }
}
