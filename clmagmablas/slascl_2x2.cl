/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @generated from clmagmablas/zlascl_2x2.cl, normal z -> s, Tue Jun 18 16:14:14 2019

       auto-converted from slascl_2x2.cu

       @author Ichitaro Yamazaki
*/
#include "kernels_header.h"
#include "slascl_2x2.h"

#define A(i_, j_) (A[(i_) + (j_)*lda])
#define W(i_, j_) (W[(i_) + (j_)*ldw])

// each thread block does one NB x n block row of A.
// each thread does one row, starting from left edge and moving right to diagonal.
__kernel void
slascl_2x2_lower(
    magma_int_t m,
    __global const float* W, unsigned long W_offset, magma_int_t ldw,
    __global float* A, unsigned long A_offset, magma_int_t lda)
{
    W += W_offset;
    A += A_offset;

    int ind = get_group_id(0) * NB + get_local_id(0);

    float D21 = W( 1, 0 );
    float D11 = MAGMA_S_DIV( W( 1, 1 ), D21 );
    float D22 = MAGMA_S_DIV( W( 0, 0 ), MAGMA_S_CNJG( D21 ) );
    float T = 1.0 / ( MAGMA_S_REAL( MAGMA_S_MUL( D11, D22 ) ) - 1.0 );
    D21 = MAGMA_S_DIV( MAGMA_S_MAKE(T,0.0), D21 );

    if (ind < m) {
        A( ind, 0 ) = MAGMA_S_MUL( MAGMA_S_CNJG( D21 ),
                                   MAGMA_S_SUB( MAGMA_S_MUL( D11, W(2+ind,0) ),
                                                W(2+ind,1) ));
        A( ind, 1 ) = MAGMA_S_MUL( D21,
                                   MAGMA_S_SUB( MAGMA_S_MUL( D22, W(2+ind,1) ),
                                                W(2+ind,0) ));
    }
}


// each thread block does one NB x n block row of A.
// each thread does one row, starting from right edge and moving left to diagonal.
__kernel void
slascl_2x2_upper(
    magma_int_t m,
    __global const float *W, unsigned long W_offset, magma_int_t ldw,
    __global float* A, unsigned long A_offset, magma_int_t lda)
{
    W += W_offset;
    A += A_offset;

    int ind = get_group_id(0) * NB + get_local_id(0);

    float D21 = W( m, 1 );
    float D11 = MAGMA_S_DIV( W( m+1, 1 ), MAGMA_S_CNJG( D21 ) );
    float D22 = MAGMA_S_DIV( W( m, 0 ), D21 );
    float T = 1.0 / ( MAGMA_S_REAL( MAGMA_S_MUL( D11, D22 ) ) - 1.0 );
    D21 = MAGMA_S_DIV( MAGMA_S_MAKE(T,0.0), D21 );

    if (ind < m) {
        A( ind, 0 ) = MAGMA_S_MUL( D21,
                                   MAGMA_S_SUB( MAGMA_S_MUL( D11, W(ind,0) ),
                                                W(ind,1) ));
        A( ind, 1 ) = MAGMA_S_MUL( MAGMA_S_CNJG( D21 ),
                                   MAGMA_S_SUB( MAGMA_S_MUL( D22, W(ind,1) ),
                                                W(ind,0) ));
    }
}
