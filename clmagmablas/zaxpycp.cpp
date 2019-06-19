/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c

       auto-converted from zaxpycp.cu

*/
#include "clmagma_runtime.h"
#include "common_magma.h"
#include "zaxpycp.h"


// ----------------------------------------------------------------------
// adds   x += r  --and--
// copies r = b
extern "C" void
magmablas_zaxpycp(
    magma_int_t m,
    magmaDoubleComplex_ptr r, size_t r_offset,
    magmaDoubleComplex_ptr x, size_t x_offset,
    magmaDoubleComplex_const_ptr b, size_t b_offset,
    magma_queue_t queue )
{
    cl_kernel kernel;
    cl_int err;
    int arg;

    const int ndim = 1;
    size_t threads[ndim];
    threads[0] = NB;
    size_t grid[ndim];
    grid[0] = magma_ceildiv( m, NB );
    grid[0] *= threads[0];
    kernel = g_runtime.get_kernel( "zaxpycp_kernel" );
    if ( kernel != NULL ) {
        err = 0;
        arg = 0;
        err |= clSetKernelArg( kernel, arg++, sizeof(m       ), &m        );
        err |= clSetKernelArg( kernel, arg++, sizeof(r       ), &r        );
        err |= clSetKernelArg( kernel, arg++, sizeof(r_offset), &r_offset );
        err |= clSetKernelArg( kernel, arg++, sizeof(x       ), &x        );
        err |= clSetKernelArg( kernel, arg++, sizeof(x_offset), &x_offset );
        err |= clSetKernelArg( kernel, arg++, sizeof(b       ), &b        );
        err |= clSetKernelArg( kernel, arg++, sizeof(b_offset), &b_offset );
        check_error( err );

        err = clEnqueueNDRangeKernel( queue, kernel, ndim, NULL, grid, threads, 0, NULL, NULL );
        check_error( err );
    }
}
