/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @generated from clmagmablas/zlat2c.cl, mixed zc -> ds, Tue Jun 18 16:14:14 2019

       auto-converted from dlat2s.cu
       @author Mark Gates
*/
#include "kernels_header.h"
#include "dlat2s.h"

#define PRECISION_d

/*
    Divides matrix into ceil( n/BLK_X ) x ceil( n/BLK_Y ) blocks.
    Each block has BLK_X threads.
    Each thread loops across one row, updating BLK_Y entries.
    Updates only the diagonal and below.
    Blocks that are fully above the diagonal exit immediately.
    
    Code similar to dlag2s and zlaset.
*/
__kernel
void dlat2s_lower(
    magma_int_t n,
    __global const double *A, unsigned long A_offset, magma_int_t lda,
    __global float *SA, unsigned long SA_offset,       magma_int_t ldsa,
    double rmax,
    __global magma_int_t *flag )
{
    A += A_offset;
    SA += SA_offset;

    double tmp;
    double neg_rmax = - rmax;
    
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
                tmp = A[j*lda];
                SA[j*ldsa] = MAGMA_S_MAKE( (float)MAGMA_D_REAL(tmp),
                                           (float)MAGMA_D_IMAG(tmp) );
                // very oddly, doing this check before assigning to SA causes
                // wrong results in dlat2s, but not zlatc -- compiler bug? -mgates
                // moving the check below fixes it (also in 3 other occurances below).
                if (   (MAGMA_D_REAL(tmp) < neg_rmax) || (MAGMA_D_REAL(tmp) > rmax)
                    #if defined(PRECISION_z) || defined(PRECISION_c)
                    || (MAGMA_D_IMAG(tmp) < neg_rmax) || (MAGMA_D_IMAG(tmp) > rmax)
                    #endif
                    )
                {
                    *flag = 1;
                }
            }
        }
        else {
            // either partial block-column or diagonal block
            for( int j=0; j < BLK_Y && iby+j < n && ind >= iby+j; ++j ) {
                tmp = A[j*lda];
                SA[j*ldsa] = MAGMA_S_MAKE( (float)MAGMA_D_REAL(tmp),
                                           (float)MAGMA_D_IMAG(tmp) );
                if (   (MAGMA_D_REAL(tmp) < neg_rmax) || (MAGMA_D_REAL(tmp) > rmax)
                    #if defined(PRECISION_z) || defined(PRECISION_c)
                    || (MAGMA_D_IMAG(tmp) < neg_rmax) || (MAGMA_D_IMAG(tmp) > rmax)
                    #endif
                    )
                {
                    *flag = 1;
                }
            }
        }
    }
}


/*
    Similar to dlat2s_full, but updates only the diagonal and above.
    Blocks that are fully below the diagonal exit immediately.
    
    Code similar to dlag2s and zlaset.
*/
__kernel
void dlat2s_upper(
    magma_int_t n,
    __global const double *A, unsigned long A_offset, magma_int_t lda,
    __global float *SA, unsigned long SA_offset,       magma_int_t ldsa,
    double rmax,
    __global magma_int_t *flag )
{
    A += A_offset;
    SA += SA_offset;

    double tmp;
    double neg_rmax = - rmax;
    
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
                tmp = A[j*lda];
                SA[j*ldsa] = MAGMA_S_MAKE( (float)MAGMA_D_REAL(tmp),
                                           (float)MAGMA_D_IMAG(tmp) );
                if (   (MAGMA_D_REAL(tmp) < neg_rmax) || (MAGMA_D_REAL(tmp) > rmax)
                    #if defined(PRECISION_z) || defined(PRECISION_c)
                    || (MAGMA_D_IMAG(tmp) < neg_rmax) || (MAGMA_D_IMAG(tmp) > rmax)
                    #endif
                    )
                {
                    *flag = 1;
                }
            }
        }
        else {
            // either partial block-column or diagonal block
            for( int j=0; j < BLK_Y && iby+j < n; ++j ) {
                if ( ind <= iby+j ) {
                    tmp = A[j*lda];
                    SA[j*ldsa] = MAGMA_S_MAKE( (float)MAGMA_D_REAL(tmp),
                                               (float)MAGMA_D_IMAG(tmp) );
                    if (   (MAGMA_D_REAL(tmp) < neg_rmax) || (MAGMA_D_REAL(tmp) > rmax)
                         #if defined(PRECISION_z) || defined(PRECISION_c)
                         || (MAGMA_D_IMAG(tmp) < neg_rmax) || (MAGMA_D_IMAG(tmp) > rmax)
                         #endif
                        )
                    {
                        *flag = 1;
                    }
                }
            }
        }
    }
}
