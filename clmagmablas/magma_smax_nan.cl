/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date
       
       @author Mark Gates
       @generated from clmagmablas/magma_dmax_nan.cl, normal d -> s, Tue Jun 18 16:14:14 2019

       auto-converted from magma_smax_nan.cu
*/
#include "kernels_header.h"
#include "magma_smax_nan.h"

// ----------------------------------------
/// Same as magma_max_reduce, but propogates nan values.
///
/// Does max reduction of n-element array x, leaving total in x[0].
/// Contents of x are destroyed in the process.
/// With k threads, can reduce array up to 2*k in size.
/// Assumes number of threads <= 1024 (which is max number of threads up to CUDA capability 3.0)
/// Calls __syncthreads before & after reduction.
/* __device__ */
void
magma_smax_nan_devfunc_n( int n, int i, __local float* x, unsigned long x_offset );

void
magma_smax_nan_devfunc_n( int n, int i, __local float* x, unsigned long x_offset )
{
    x += x_offset;

    barrier( CLK_LOCAL_MEM_FENCE );
    //if ( n > 1024 ) { if ( i < 1024 && i + 1024 < n ) { x[i] = max_nan( x[i], x[i+1024] ); }  barrier( CLK_LOCAL_MEM_FENCE ); }
    //if ( n >  512 ) { if ( i <  512 && i +  512 < n ) { x[i] = max_nan( x[i], x[i+ 512] ); }  barrier( CLK_LOCAL_MEM_FENCE ); }
    if ( n >  256 ) { if ( i <  256 && i +  256 < n ) { x[i] = max_nan( x[i], x[i+ 256] ); }  barrier( CLK_LOCAL_MEM_FENCE ); }
    if ( n >  128 ) { if ( i <  128 && i +  128 < n ) { x[i] = max_nan( x[i], x[i+ 128] ); }  barrier( CLK_LOCAL_MEM_FENCE ); }
    if ( n >   64 ) { if ( i <   64 && i +   64 < n ) { x[i] = max_nan( x[i], x[i+  64] ); }  barrier( CLK_LOCAL_MEM_FENCE ); }
    if ( n >   32 ) { if ( i <   32 && i +   32 < n ) { x[i] = max_nan( x[i], x[i+  32] ); }  barrier( CLK_LOCAL_MEM_FENCE ); }
    if ( n >   16 ) { if ( i <   16 && i +   16 < n ) { x[i] = max_nan( x[i], x[i+  16] ); }  barrier( CLK_LOCAL_MEM_FENCE ); }
    if ( n >    8 ) { if ( i <    8 && i +    8 < n ) { x[i] = max_nan( x[i], x[i+   8] ); }  barrier( CLK_LOCAL_MEM_FENCE ); }
    if ( n >    4 ) { if ( i <    4 && i +    4 < n ) { x[i] = max_nan( x[i], x[i+   4] ); }  barrier( CLK_LOCAL_MEM_FENCE ); }
    if ( n >    2 ) { if ( i <    2 && i +    2 < n ) { x[i] = max_nan( x[i], x[i+   2] ); }  barrier( CLK_LOCAL_MEM_FENCE ); }
    if ( n >    1 ) { if ( i <    1 && i +    1 < n ) { x[i] = max_nan( x[i], x[i+   1] ); }  barrier( CLK_LOCAL_MEM_FENCE ); }
}
// end max_nan_reduce


// ----------------------------------------
/// max reduction, for arbitrary size vector. Leaves max(x) in x[0].
/// Uses only one thread block of 512 threads, so is not efficient for really large vectors.
__kernel void
magma_smax_nan_kernel( magma_int_t n, __global float* x, unsigned long x_offset )
{
    x += x_offset;
    
    __local float smax[ NB ];
    int tx = get_local_id(0);
    
    smax[tx] = 0;
    for( int i=tx; i < n; i += NB ) {
        smax[tx] = max_nan( smax[tx], x[i] );
    }
    magma_smax_nan_devfunc_n( NB, tx, smax, 0 );
    if ( tx == 0 ) {
        x[0] = smax[0];
    }
}
