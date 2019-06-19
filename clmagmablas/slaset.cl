/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Mark Gates
       @author Azzam Haidar
       
       @generated from clmagmablas/zlaset.cl, normal z -> s, Tue Jun 18 16:14:14 2019

       auto-converted from slaset.cu

*/
#include "kernels_header.h"
#include "slaset.h"

/*
    Divides matrix into ceil( m/BLK_X ) x ceil( n/BLK_Y ) blocks.
    Each block has BLK_X threads.
    Each thread loops across one row, updating BLK_Y entries.

    Code similar to slaset, slacpy, slag2d, clag2z, sgeadd.
*/

// prototype to suppress compiler warning
void slaset_full_device(
    magma_int_t m, magma_int_t n,
    float offdiag, float diag,
    __global float *A, magma_int_t lda );

void slaset_full_device(
    magma_int_t m, magma_int_t n,
    float offdiag, float diag,
    __global float *A, magma_int_t lda )
{
    int ind = get_group_id(0)*BLK_X + get_local_id(0);
    int iby = get_group_id(1)*BLK_Y;
    /* check if full block-column && (below diag || above diag || offdiag == diag) */
    bool full = (iby + BLK_Y <= n && (ind >= iby + BLK_Y || ind + BLK_X <= iby || MAGMA_S_EQUAL( offdiag, diag )));
    /* do only rows inside matrix */
    if ( ind < m ) {
        A += ind + iby*lda;
        if ( full ) {
            // full block-column, off-diagonal block or offdiag == diag
            #pragma unroll
            for( int j=0; j < BLK_Y; ++j ) {
                A[j*lda] = offdiag;
            }
        }
        else {
            // either partial block-column or diagonal block
            for( int j=0; j < BLK_Y && iby+j < n; ++j ) {
                if ( iby+j == ind )
                    A[j*lda] = diag;
                else
                    A[j*lda] = offdiag;
            }
        }
    }
}


/*
    Similar to slaset_full, but updates only the diagonal and below.
    Blocks that are fully above the diagonal exit immediately.

    Code similar to slaset, slacpy, zlat2c, clat2z.
*/

// prototype to suppress compiler warning
void slaset_lower_device(
    magma_int_t m, magma_int_t n,
    float offdiag, float diag,
    __global float *A, magma_int_t lda );

void slaset_lower_device(
    magma_int_t m, magma_int_t n,
    float offdiag, float diag,
    __global float *A, magma_int_t lda )
{
    int ind = get_group_id(0)*BLK_X + get_local_id(0);
    int iby = get_group_id(1)*BLK_Y;
    /* check if full block-column && (below diag) */
    bool full = (iby + BLK_Y <= n && (ind >= iby + BLK_Y));
    /* do only rows inside matrix, and blocks not above diag */
    if ( ind < m && ind + BLK_X > iby ) {
        A += ind + iby*lda;
        if ( full ) {
            // full block-column, off-diagonal block
            #pragma unroll
            for( int j=0; j < BLK_Y; ++j ) {
                A[j*lda] = offdiag;
            }
        }
        else {
            // either partial block-column or diagonal block
            for( int j=0; j < BLK_Y && iby+j < n; ++j ) {
                if ( iby+j == ind )
                    A[j*lda] = diag;
                else if ( ind > iby+j )
                    A[j*lda] = offdiag;
            }
        }
    }
}


/*
    Similar to slaset_full, but updates only the diagonal and above.
    Blocks that are fully below the diagonal exit immediately.

    Code similar to slaset, slacpy, zlat2c, clat2z.
*/

// prototype to suppress compiler warning
void slaset_upper_device(
    magma_int_t m, magma_int_t n,
    float offdiag, float diag,
    __global float *A, magma_int_t lda );

void slaset_upper_device(
    magma_int_t m, magma_int_t n,
    float offdiag, float diag,
    __global float *A, magma_int_t lda )
{
    int ind = get_group_id(0)*BLK_X + get_local_id(0);
    int iby = get_group_id(1)*BLK_Y;
    /* check if full block-column && (above diag) */
    bool full = (iby + BLK_Y <= n && (ind + BLK_X <= iby));
    /* do only rows inside matrix, and blocks not below diag */
    if ( ind < m && ind < iby + BLK_Y ) {
        A += ind + iby*lda;
        if ( full ) {
            // full block-column, off-diagonal block
            #pragma unroll
            for( int j=0; j < BLK_Y; ++j ) {
                A[j*lda] = offdiag;
            }
        }
        else {
            // either partial block-column or diagonal block
            for( int j=0; j < BLK_Y && iby+j < n; ++j ) {
                if ( iby+j == ind )
                    A[j*lda] = diag;
                else if ( ind < iby+j )
                    A[j*lda] = offdiag;
            }
        }
    }
}


//////////////////////////////////////////////////////////////////////////////////////
/*
    kernel wrappers to call the device functions.
*/
__kernel
void slaset_full_kernel(
    magma_int_t m, magma_int_t n,
    float offdiag, float diag,
    __global float *dA, unsigned long dA_offset, magma_int_t ldda )
{
    dA += dA_offset;

    slaset_full_device(m, n, offdiag, diag, dA, ldda);
}

__kernel
void slaset_lower_kernel(
    magma_int_t m, magma_int_t n,
    float offdiag, float diag,
    __global float *dA, unsigned long dA_offset, magma_int_t ldda )
{
    dA += dA_offset;

    slaset_lower_device(m, n, offdiag, diag, dA, ldda);
}

__kernel
void slaset_upper_kernel(
    magma_int_t m, magma_int_t n,
    float offdiag, float diag,
    __global float *dA, unsigned long dA_offset, magma_int_t ldda )
{
    dA += dA_offset;

    slaset_upper_device(m, n, offdiag, diag, dA, ldda);
}
