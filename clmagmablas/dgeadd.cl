/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @generated from clmagmablas/zgeadd.cl, normal z -> d, Tue Jun 18 16:14:14 2019

       auto-converted from dgeadd.cu
       @author Mark Gates
*/
#include "kernels_header.h"
#include "dgeadd.h"

/*
    Divides matrix into ceil( m/BLK_X ) x ceil( n/BLK_Y ) blocks.
    Each block has BLK_X threads.
    Each thread loops across one row, updating BLK_Y entries.

    Code similar to dlaset.
*/
__kernel
void dgeadd_full(
    magma_int_t m, magma_int_t n,
    double alpha,
    __global const double *dA, unsigned long dA_offset, magma_int_t ldda,
    __global double       *dB, unsigned long dB_offset, magma_int_t lddb )
{
    dA += dA_offset;
    dB += dB_offset;

    int ind = get_group_id(0)*BLK_X + get_local_id(0);
    int iby = get_group_id(1)*BLK_Y;
    /* check if full block-column */
    bool full = (iby + BLK_Y <= n);
    /* do only rows inside matrix */
    if ( ind < m ) {
        dA += ind + iby*ldda;
        dB += ind + iby*lddb;
        if ( full ) {
            // full block-column
            #pragma unroll
            for( int j=0; j < BLK_Y; ++j ) {
                dB[j*lddb] = MAGMA_D_ADD( MAGMA_D_MUL( alpha, dA[j*ldda] ), dB[j*lddb] );
            }
        }
        else {
            // partial block-column
            for( int j=0; j < BLK_Y && iby+j < n; ++j ) {
                dB[j*lddb] = MAGMA_D_ADD( MAGMA_D_MUL( alpha, dA[j*ldda] ), dB[j*lddb] );
            }
        }
    }
}
