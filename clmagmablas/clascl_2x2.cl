/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @generated from clmagmablas/zlascl_2x2.cl, normal z -> c, Tue Jun 18 16:14:14 2019

       auto-converted from clascl_2x2.cu

       @author Ichitaro Yamazaki
*/
#include "kernels_header.h"
#include "clascl_2x2.h"

#define A(i_, j_) (A[(i_) + (j_)*lda])
#define W(i_, j_) (W[(i_) + (j_)*ldw])

// each thread block does one NB x n block row of A.
// each thread does one row, starting from left edge and moving right to diagonal.
__kernel void
clascl_2x2_lower(
    magma_int_t m,
    __global const magmaFloatComplex* W, unsigned long W_offset, magma_int_t ldw,
    __global magmaFloatComplex* A, unsigned long A_offset, magma_int_t lda)
{
    W += W_offset;
    A += A_offset;

    int ind = get_group_id(0) * NB + get_local_id(0);

    magmaFloatComplex D21 = W( 1, 0 );
    magmaFloatComplex D11 = MAGMA_C_DIV( W( 1, 1 ), D21 );
    magmaFloatComplex D22 = MAGMA_C_DIV( W( 0, 0 ), MAGMA_C_CNJG( D21 ) );
    float T = 1.0 / ( MAGMA_C_REAL( MAGMA_C_MUL( D11, D22 ) ) - 1.0 );
    D21 = MAGMA_C_DIV( MAGMA_C_MAKE(T,0.0), D21 );

    if (ind < m) {
        A( ind, 0 ) = MAGMA_C_MUL( MAGMA_C_CNJG( D21 ),
                                   MAGMA_C_SUB( MAGMA_C_MUL( D11, W(2+ind,0) ),
                                                W(2+ind,1) ));
        A( ind, 1 ) = MAGMA_C_MUL( D21,
                                   MAGMA_C_SUB( MAGMA_C_MUL( D22, W(2+ind,1) ),
                                                W(2+ind,0) ));
    }
}


// each thread block does one NB x n block row of A.
// each thread does one row, starting from right edge and moving left to diagonal.
__kernel void
clascl_2x2_upper(
    magma_int_t m,
    __global const magmaFloatComplex *W, unsigned long W_offset, magma_int_t ldw,
    __global magmaFloatComplex* A, unsigned long A_offset, magma_int_t lda)
{
    W += W_offset;
    A += A_offset;

    int ind = get_group_id(0) * NB + get_local_id(0);

    magmaFloatComplex D21 = W( m, 1 );
    magmaFloatComplex D11 = MAGMA_C_DIV( W( m+1, 1 ), MAGMA_C_CNJG( D21 ) );
    magmaFloatComplex D22 = MAGMA_C_DIV( W( m, 0 ), D21 );
    float T = 1.0 / ( MAGMA_C_REAL( MAGMA_C_MUL( D11, D22 ) ) - 1.0 );
    D21 = MAGMA_C_DIV( MAGMA_C_MAKE(T,0.0), D21 );

    if (ind < m) {
        A( ind, 0 ) = MAGMA_C_MUL( D21,
                                   MAGMA_C_SUB( MAGMA_C_MUL( D11, W(ind,0) ),
                                                W(ind,1) ));
        A( ind, 1 ) = MAGMA_C_MUL( MAGMA_C_CNJG( D21 ),
                                   MAGMA_C_SUB( MAGMA_C_MUL( D22, W(ind,1) ),
                                                W(ind,0) ));
    }
}