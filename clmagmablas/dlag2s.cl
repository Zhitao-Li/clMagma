/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @generated from clmagmablas/zlag2c.cl, mixed zc -> ds, Tue Jun 18 16:14:14 2019

       auto-converted from dlag2s.cu
       @author Mark Gates
*/
#include "kernels_header.h"
#include "dlag2s.h"

#define PRECISION_d


// TODO get rid of global variable!
// TODO __kernel int flag = 0;


/*
    Divides matrix into ceil( m/BLK_X ) x ceil( n/BLK_Y ) blocks.
    Each block has BLK_X threads.
    Each thread loops across one row, updating BLK_Y entries.
    
    Code similar to dlat2s and zlaset.
*/
__kernel
void dlag2s_kernel(
    magma_int_t m, magma_int_t n,
    __global const double *A, unsigned long A_offset, magma_int_t lda,
    __global float *SA, unsigned long SA_offset,      magma_int_t ldsa,
    double rmax )
{
    A += A_offset;
    SA += SA_offset;

    double tmp;
    // double neg_rmax = - rmax;
    
    int ind = get_group_id(0)*BLK_X + get_local_id(0);
    int iby = get_group_id(1)*BLK_Y;
    /* check if full block-column */
    bool full = (iby + BLK_Y <= n);
    /* do only rows inside matrix */
    if ( ind < m ) {
        A  += ind + iby*lda;
        SA += ind + iby*ldsa;
        if ( full ) {
            // full block-column
            #pragma unroll
            for( int j=0; j < BLK_Y; ++j ) {
                tmp = A[j*lda];
                // if (   (MAGMA_D_REAL(tmp) < neg_rmax) || (MAGMA_D_REAL(tmp) > rmax)
                //     #if defined(PRECISION_z) || defined(PRECISION_c)
                //     || (MAGMA_D_IMAG(tmp) < neg_rmax) || (MAGMA_D_IMAG(tmp) > rmax)
                //     #endif
                //     )
                // {
                //     flag = 1;
                // }
                SA[j*ldsa] = MAGMA_S_MAKE( MAGMA_D_REAL(tmp), MAGMA_D_IMAG(tmp) );
            }
        }
        else {
            // partial block-column
            for( int j=0; j < BLK_Y && iby+j < n; ++j ) {
                tmp = A[j*lda];
                // if (   (MAGMA_D_REAL(tmp) < neg_rmax) || (MAGMA_D_REAL(tmp) > rmax)
                //     #if defined(PRECISION_z) || defined(PRECISION_c)
                //     || (MAGMA_D_IMAG(tmp) < neg_rmax) || (MAGMA_D_IMAG(tmp) > rmax)
                //     #endif
                //     )
                // {
                //     flag = 1;
                // }
                SA[j*ldsa] = MAGMA_S_MAKE( MAGMA_D_REAL(tmp), MAGMA_D_IMAG(tmp) );
            }
        }
    }
}
