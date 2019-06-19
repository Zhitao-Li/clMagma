/*
 *   -- clMAGMA (version 0.4) --
 *      Univ. of Tennessee, Knoxville
 *      Univ. of California, Berkeley
 *      Univ. of Colorado, Denver
 *      @date
 */
#include "clmagma_runtime.h"
#include "common_magma.h"


#define BLOCK_SIZE 64


// empty kernel calling, benchmarkde for overhead for iwocl 2013
// (updated to current formatting standards, no precision generation)
extern "C" void
magmablas_empty(
    magmaDouble_ptr dA,
    magmaDouble_ptr dB,
    magmaDouble_ptr dC,
    magma_queue_t queue )
{
    cl_kernel kernel;
    cl_int err;
    int arg;
    
    int n=1;
    magma_int_t i0=0, i1=1, i2=1, i3=1, i4=1, i5=1, i6=1, i7=1, i8=1, i9=1;
    double d0=1, d1=1, d2=1, d3=1, d4=1;

    size_t threads[1];
    threads[0] = BLOCK_SIZE;
    size_t grid[1];
    grid[0] = magma_ceildiv( n, BLOCK_SIZE );
    grid[0] *= threads[0];
    
    kernel = g_runtime.get_kernel( "empty_kernel" );
    if ( kernel != NULL ) {
        err = 0;
        arg = 0;
        err  = clSetKernelArg( kernel, arg++, sizeof(i0), &i0 );
        err |= clSetKernelArg( kernel, arg++, sizeof(i1), &i1 );
        err |= clSetKernelArg( kernel, arg++, sizeof(i2), &i2 );
        err |= clSetKernelArg( kernel, arg++, sizeof(i3), &i3 );
        err |= clSetKernelArg( kernel, arg++, sizeof(i4), &i4 );
        err |= clSetKernelArg( kernel, arg++, sizeof(i5), &i5 );
        err |= clSetKernelArg( kernel, arg++, sizeof(i6), &i6 );
        err |= clSetKernelArg( kernel, arg++, sizeof(i7), &i7 );
        err |= clSetKernelArg( kernel, arg++, sizeof(i8), &i8 );
        err |= clSetKernelArg( kernel, arg++, sizeof(i9), &i9 );
        
        err |= clSetKernelArg( kernel, arg++, sizeof(d0), &d0 );
        err |= clSetKernelArg( kernel, arg++, sizeof(d1), &d1 );
        err |= clSetKernelArg( kernel, arg++, sizeof(d2), &d2 );
        err |= clSetKernelArg( kernel, arg++, sizeof(d3), &d3 );
        err |= clSetKernelArg( kernel, arg++, sizeof(d4), &d4 );
        
        err |= clSetKernelArg( kernel, arg++, sizeof(dA), &dA );
        err |= clSetKernelArg( kernel, arg++, sizeof(dB), &dB );
        err |= clSetKernelArg( kernel, arg++, sizeof(dC), &dC );
        check_error( err );
        
        err = clEnqueueNDRangeKernel( queue, kernel, 1, NULL, grid, threads, 0, NULL, NULL );
        check_error( err );
    }
}
