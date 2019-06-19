/*
 *   -- clMAGMA (version 0.4) --
 *      Univ. of Tennessee, Knoxville
 *      Univ. of California, Berkeley
 *      Univ. of Colorado, Denver
 *      @date
 */
#include "kernels_header.h"

// empty kernel, benchmark in iwocl 2013
__kernel void empty_kernel(
    magma_int_t i0, magma_int_t i1, magma_int_t i2, magma_int_t i3, magma_int_t i4, 
    magma_int_t i5, magma_int_t i6, magma_int_t i7, magma_int_t i8, magma_int_t i9,
    double d0, double d1, double d2, double d3, double d4, 
    __global double *dA,
    __global double *dB,
    __global double *dC )
{
    int tid = get_local_id(0);

    for( int i=0; i < i0; i++ ) {
        dC[i+tid] += d1*dC[i+tid] + d2*dA[i]*dB[i];
    }
    barrier( CLK_LOCAL_MEM_FENCE );
    for( int i=0; i < i0; i++ ) {
        dC[i+tid] += d1*dC[i+tid] + d2*dA[i]*dB[i];
    }
}
