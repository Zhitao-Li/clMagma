/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @generated from clmagmablas/zlascl_diag.cl, normal z -> c, Tue Jun 18 16:14:14 2019

       auto-converted from clascl_diag.cu
*/
#include "kernels_header.h"
#include "clascl_diag.h"


// each thread block does one NB x n block row of A.
// each thread does one row, starting from left edge and moving right to diagonal.
__kernel void
clascl_diag_lower(
    magma_int_t m, magma_int_t n,
    __global const magmaFloatComplex* D, unsigned long D_offset, magma_int_t ldd,
    __global magmaFloatComplex*       A, unsigned long A_offset, magma_int_t lda)
{
    D += D_offset;
    A += A_offset;

    int ind = get_group_id(0) * NB + get_local_id(0);

    A += ind;
    if (ind < m) {
        for (int j=0; j < n; j++ ) {
            A[j*lda] = MAGMA_C_DIV( A[j*lda], D[j + j*ldd] );
        }
    }
}


// each thread block does one NB x n block row of A.
// each thread does one row, starting from right edge and moving left to diagonal.
__kernel void
clascl_diag_upper(
    magma_int_t m, magma_int_t n,
    __global const magmaFloatComplex* D, unsigned long D_offset, magma_int_t ldd,
    __global magmaFloatComplex*       A, unsigned long A_offset, magma_int_t lda)
{
    D += D_offset;
    A += A_offset;

    int ind = get_group_id(0) * NB + get_local_id(0);

    A += ind;
    if (ind < m) {
        for (int j=0; j < n; j++ ) {
            A[j*lda] = MAGMA_C_DIV( A[j*lda], D[ind + ind*ldd] );
        }
    }
}
