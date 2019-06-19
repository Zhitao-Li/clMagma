/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Mark Gates
       @author Azzam Haidar
       
       @generated from clmagmablas/zlacpy.cl, normal z -> c, Tue Jun 18 16:14:14 2019

       auto-converted from clacpy.cu

*/
#include "kernels_header.h"
#include "clacpy.h"

/*
    Divides matrix into ceil( m/BLK_X ) x ceil( n/BLK_Y ) blocks.
    Each block has BLK_X threads.
    Each thread loops across one row, updating BLK_Y entries.

    Code similar to claset, clacpy, clag2z, clag2z, cgeadd.
*/

// prototype to suppress compiler warning
void clacpy_full_device(
    magma_int_t m, magma_int_t n,
    __global const magmaFloatComplex *dA, magma_int_t ldda,
    __global magmaFloatComplex       *dB, magma_int_t lddb );

void clacpy_full_device(
    magma_int_t m, magma_int_t n,
    __global const magmaFloatComplex *dA, magma_int_t ldda,
    __global magmaFloatComplex       *dB, magma_int_t lddb )
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
    Similar to clacpy_full, but updates only the diagonal and below.
    Blocks that are fully above the diagonal exit immediately.

    Code similar to claset, clacpy, zlat2c, clat2z.
*/

// prototype to suppress compiler warning
void clacpy_lower_device(
    magma_int_t m, magma_int_t n,
    __global const magmaFloatComplex *dA, magma_int_t ldda,
    __global magmaFloatComplex       *dB, magma_int_t lddb );

void clacpy_lower_device(
    magma_int_t m, magma_int_t n,
    __global const magmaFloatComplex *dA, magma_int_t ldda,
    __global magmaFloatComplex       *dB, magma_int_t lddb )
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
    Similar to clacpy_full, but updates only the diagonal and above.
    Blocks that are fully below the diagonal exit immediately.

    Code similar to claset, clacpy, zlat2c, clat2z.
*/

// prototype to suppress compiler warning
void clacpy_upper_device(
    magma_int_t m, magma_int_t n,
    __global const magmaFloatComplex *dA, magma_int_t ldda,
    __global magmaFloatComplex       *dB, magma_int_t lddb );

void clacpy_upper_device(
    magma_int_t m, magma_int_t n,
    __global const magmaFloatComplex *dA, magma_int_t ldda,
    __global magmaFloatComplex       *dB, magma_int_t lddb )
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
void clacpy_full_kernel(
    magma_int_t m, magma_int_t n,
    __global const magmaFloatComplex *dA, unsigned long dA_offset, magma_int_t ldda,
    __global magmaFloatComplex       *dB, unsigned long dB_offset, magma_int_t lddb )
{
    dA += dA_offset;
    dB += dB_offset;

    clacpy_full_device(m, n, dA, ldda, dB, lddb);
}

__kernel
void clacpy_lower_kernel(
    magma_int_t m, magma_int_t n,
    __global const magmaFloatComplex *dA, unsigned long dA_offset, magma_int_t ldda,
    __global magmaFloatComplex       *dB, unsigned long dB_offset, magma_int_t lddb )
{
    dA += dA_offset;
    dB += dB_offset;

    clacpy_lower_device(m, n, dA, ldda, dB, lddb);
}

__kernel
void clacpy_upper_kernel(
    magma_int_t m, magma_int_t n,
    __global const magmaFloatComplex *dA, unsigned long dA_offset, magma_int_t ldda,
    __global magmaFloatComplex       *dB, unsigned long dB_offset, magma_int_t lddb )
{
    dA += dA_offset;
    dB += dB_offset;

    clacpy_upper_device(m, n, dA, ldda, dB, lddb);
}
