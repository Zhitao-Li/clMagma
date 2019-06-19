/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Mark Gates
       @author Azzam Haidar
       
       @generated from clmagmablas/zlacpy.cl, normal z -> s, Tue Jun 18 16:14:14 2019

       auto-converted from slacpy.cu

*/
#include "kernels_header.h"
#include "slacpy.h"

/*
    Divides matrix into ceil( m/BLK_X ) x ceil( n/BLK_Y ) blocks.
    Each block has BLK_X threads.
    Each thread loops across one row, updating BLK_Y entries.

    Code similar to slaset, slacpy, slag2d, clag2z, sgeadd.
*/

// prototype to suppress compiler warning
void slacpy_full_device(
    magma_int_t m, magma_int_t n,
    __global const float *dA, magma_int_t ldda,
    __global float       *dB, magma_int_t lddb );

void slacpy_full_device(
    magma_int_t m, magma_int_t n,
    __global const float *dA, magma_int_t ldda,
    __global float       *dB, magma_int_t lddb )
{
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
                dB[j*lddb] = dA[j*ldda];
            }
        }
        else {
            // partial block-column
            for( int j=0; j < BLK_Y && iby+j < n; ++j ) {
                dB[j*lddb] = dA[j*ldda];
            }
        }
    }
}


/*
    Similar to slacpy_full, but updates only the diagonal and below.
    Blocks that are fully above the diagonal exit immediately.

    Code similar to slaset, slacpy, zlat2c, clat2z.
*/

// prototype to suppress compiler warning
void slacpy_lower_device(
    magma_int_t m, magma_int_t n,
    __global const float *dA, magma_int_t ldda,
    __global float       *dB, magma_int_t lddb );

void slacpy_lower_device(
    magma_int_t m, magma_int_t n,
    __global const float *dA, magma_int_t ldda,
    __global float       *dB, magma_int_t lddb )
{
    int ind = get_group_id(0)*BLK_X + get_local_id(0);
    int iby = get_group_id(1)*BLK_Y;
    /* check if full block-column && (below diag) */
    bool full = (iby + BLK_Y <= n && (ind >= iby + BLK_Y));
    /* do only rows inside matrix, and blocks not above diag */
    if ( ind < m && ind + BLK_X > iby ) {
        dA += ind + iby*ldda;
        dB += ind + iby*lddb;
        if ( full ) {
            // full block-column, off-diagonal block
            #pragma unroll
            for( int j=0; j < BLK_Y; ++j ) {
                dB[j*lddb] = dA[j*ldda];
            }
        }
        else {
            // either partial block-column or diagonal block
            for( int j=0; j < BLK_Y && iby+j < n && ind >= iby+j; ++j ) {
                dB[j*lddb] = dA[j*ldda];
            }
        }
    }
}


/*
    Similar to slacpy_full, but updates only the diagonal and above.
    Blocks that are fully below the diagonal exit immediately.

    Code similar to slaset, slacpy, zlat2c, clat2z.
*/

// prototype to suppress compiler warning
void slacpy_upper_device(
    magma_int_t m, magma_int_t n,
    __global const float *dA, magma_int_t ldda,
    __global float       *dB, magma_int_t lddb );

void slacpy_upper_device(
    magma_int_t m, magma_int_t n,
    __global const float *dA, magma_int_t ldda,
    __global float       *dB, magma_int_t lddb )
{
    int ind = get_group_id(0)*BLK_X + get_local_id(0);
    int iby = get_group_id(1)*BLK_Y;
    /* check if full block-column && (above diag) */
    bool full = (iby + BLK_Y <= n && (ind + BLK_X <= iby));
    /* do only rows inside matrix, and blocks not below diag */
    if ( ind < m && ind < iby + BLK_Y ) {
        dA += ind + iby*ldda;
        dB += ind + iby*lddb;
        if ( full ) {
            // full block-column, off-diagonal block
            #pragma unroll
            for( int j=0; j < BLK_Y; ++j ) {
                dB[j*lddb] = dA[j*ldda];
            }
        }
        else {
            // either partial block-column or diagonal block
            for( int j=0; j < BLK_Y && iby+j < n; ++j ) {
                if ( ind <= iby+j ) {
                    dB[j*lddb] = dA[j*ldda];
                }
            }
        }
    }
}


//////////////////////////////////////////////////////////////////////////////////////
/*
    kernel wrappers to call the device functions.
*/
__kernel
void slacpy_full_kernel(
    magma_int_t m, magma_int_t n,
    __global const float *dA, unsigned long dA_offset, magma_int_t ldda,
    __global float       *dB, unsigned long dB_offset, magma_int_t lddb )
{
    dA += dA_offset;
    dB += dB_offset;

    slacpy_full_device(m, n, dA, ldda, dB, lddb);
}

__kernel
void slacpy_lower_kernel(
    magma_int_t m, magma_int_t n,
    __global const float *dA, unsigned long dA_offset, magma_int_t ldda,
    __global float       *dB, unsigned long dB_offset, magma_int_t lddb )
{
    dA += dA_offset;
    dB += dB_offset;

    slacpy_lower_device(m, n, dA, ldda, dB, lddb);
}

__kernel
void slacpy_upper_kernel(
    magma_int_t m, magma_int_t n,
    __global const float *dA, unsigned long dA_offset, magma_int_t ldda,
    __global float       *dB, unsigned long dB_offset, magma_int_t lddb )
{
    dA += dA_offset;
    dB += dB_offset;

    slacpy_upper_device(m, n, dA, ldda, dB, lddb);
}
