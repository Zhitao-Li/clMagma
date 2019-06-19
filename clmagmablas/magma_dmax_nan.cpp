/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date
       
       @author Mark Gates
       @precisions normal d -> s

       auto-converted from magma_dmax_nan.cu
*/
#include "clmagma_runtime.h"
#include "common_magma.h"
#include "magma_dmax_nan.h"

// ----------------------------------------
extern "C"
double
magmablas_dmax_nan(
    magma_int_t n,
    magmaDouble_ptr dx, size_t dx_offset,
    magma_queue_t queue )
{
    cl_kernel kernel;
    cl_int err;
    int arg;

    size_t grid[1] = { 1 };
    size_t threads[1] = { NB };
    grid[0] *= threads[0];
    kernel = g_runtime.get_kernel( "magma_dmax_nan_kernel" );
    if ( kernel != NULL ) {
        err = 0;
        arg = 0;
        err |= clSetKernelArg( kernel, arg++, sizeof(n        ), &n         );
        err |= clSetKernelArg( kernel, arg++, sizeof(dx       ), &dx        );
        err |= clSetKernelArg( kernel, arg++, sizeof(dx_offset), &dx_offset );
        check_error( err );

        err = clEnqueueNDRangeKernel( queue, kernel, 1, NULL, grid, threads, 0, NULL, NULL );
        check_error( err );
    }
    
    double res = 0;
    magma_dgetvector( 1, dx, 0, 1, &res, 1, queue );
    return res;
}
