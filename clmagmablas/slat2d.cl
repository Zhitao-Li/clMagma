/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @generated from clmagmablas/clat2z.cl, mixed zc -> ds, Tue Jun 18 16:14:14 2019

       auto-converted from slat2d.cu
       @author Mark Gates
*/
#include "kernels_header.h"
#include "slat2d.h"


/*
    Divides matrix into ceil( n/BLK_X ) x ceil( n/BLK_Y ) blocks.
    Each block has BLK_X threads.
    Each thread loops across one row, updating BLK_Y entries.
    Updates only the diagonal and below.
    Blocks that are fully above the diagonal exit immediately.
    
    Code similar to dlag2s and zlaset.
*/
__kernel
void slat2d_lower(
    magma_int_t n,
    __global const float *SA, unsigned long SA_offset, magma_int_t ldsa,
    __global double       *A, unsigned long A_offset,  magma_int_t lda )
{
    SA += SA_offset;
    A += A_offset;

    int ind = get_group_id(0)*BLK_X + get_local_id(0);
    int iby = get_group_id(1)*BLK_Y;
    /* check if full block-column && (below diag) */
    bool full = (iby + BLK_Y <= n && (ind >= iby + BLK_Y));
    /* do only rows inside matrix, and blocks not above diag */
    if ( ind < n && ind + BLK_X > iby ) {
        A  += ind + iby*lda;
        SA += ind + iby*ldsa;
        if ( full ) {
            // full block-column, off-diagonal block
            #pragma unroll
            for( int j=0; j < BLK_Y; ++j ) {
                A[j*lda] = MAGMA_D_MAKE( MAGMA_S_REAL( SA[j*ldsa] ),
                                         MAGMA_S_IMAG( SA[j*ldsa] ) );
            }
        }
        else {
            // either partial block-column or diagonal block
            for( int j=0; j < BLK_Y && iby+j < n && ind >= iby+j; ++j ) {
                A[j*lda] = MAGMA_D_MAKE( MAGMA_S_REAL( SA[j*ldsa] ),
                                         MAGMA_S_IMAG( SA[j*ldsa] ) );
            }
        }
    }
}


/*
    Similar to slat2d_full, but updates only the diagonal and above.
    Blocks that are fully below the diagonal exit immediately.
    
    Code similar to dlag2s and zlaset.
*/
__kernel
void slat2d_upper(
    magma_int_t n,
    __global const float *SA, unsigned long SA_offset, magma_int_t ldsa,
    __global double       *A, unsigned long A_offset,  magma_int_t lda )
{
    SA += SA_offset;
    A += A_offset;

    int ind = get_group_id(0)*BLK_X + get_local_id(0);
    int iby = get_group_id(1)*BLK_Y;
    /* check if full block-column && (above diag) */
    bool full = (iby + BLK_Y <= n && (ind + BLK_X <= iby));
    /* do only rows inside matrix, and blocks not below diag */
    if ( ind < n && ind < iby + BLK_Y ) {
        A  += ind + iby*lda;
        SA += ind + iby*ldsa;
        if ( full ) {
            // full block-column, off-diagonal block
            #pragma unroll
            for( int j=0; j < BLK_Y; ++j ) {
                A[j*lda] = MAGMA_D_MAKE( MAGMA_S_REAL( SA[j*ldsa] ),
                                         MAGMA_S_IMAG( SA[j*ldsa] ) );
            }
        }
        else {
            // either partial block-column or diagonal block
            for( int j=0; j < BLK_Y && iby+j < n; ++j ) {
                if ( ind <= iby+j ) {
                    A[j*lda] = MAGMA_D_MAKE( MAGMA_S_REAL( SA[j*ldsa] ),
                                             MAGMA_S_IMAG( SA[j*ldsa] ) );
                }
            }
        }
    }
}
