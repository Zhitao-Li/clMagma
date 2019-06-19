/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date
       
       @author Mark Gates

       @generated from clmagmablas/zswap.cpp, normal z -> c, Tue Jun 18 16:14:18 2019

       auto-converted from cswap.cu

*/
#include "clmagma_runtime.h"
#include "common_magma.h"
#include "cswap.h"


/**
    Purpose:
    =============
    Swap vector x and y; \f$ x <-> y \f$.

    @param[in]
    n       Number of elements in vector x and y. n >= 0.

    @param[in,out]
    dx      COMPLEX array on GPU device.
            The n element vector x of dimension (1 + (n-1)*incx).

    @param[in]
    incx    Stride between consecutive elements of dx. incx != 0.

    @param[in,out]
    dy      COMPLEX array on GPU device.
            The n element vector y of dimension (1 + (n-1)*incy).

    @param[in]
    incy    Stride between consecutive elements of dy. incy != 0.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_cblas1
    ********************************************************************/
extern "C" void 
magmablas_cswap(
    magma_int_t n,
    magmaFloatComplex_ptr dx, size_t dx_offset, magma_int_t incx, 
    magmaFloatComplex_ptr dy, size_t dy_offset, magma_int_t incy,
    magma_queue_t queue )
{
    cl_kernel kernel;
    cl_int err;
    int arg;

    const int ndim = 1;
    size_t threads[ndim];
    threads[0] = NB;
    size_t grid[ndim];
    grid[0] = magma_ceildiv( n, NB );
    grid[0] *= threads[0];
    kernel = g_runtime.get_kernel( "cswap_kernel" );
    if ( kernel != NULL ) {
        err = 0;
        arg = 0;
        err |= clSetKernelArg( kernel, arg++, sizeof(n        ), &n         );
        err |= clSetKernelArg( kernel, arg++, sizeof(dx       ), &dx        );
        err |= clSetKernelArg( kernel, arg++, sizeof(dx_offset), &dx_offset );
        err |= clSetKernelArg( kernel, arg++, sizeof(incx     ), &incx      );
        err |= clSetKernelArg( kernel, arg++, sizeof(dy       ), &dy        );
        err |= clSetKernelArg( kernel, arg++, sizeof(dy_offset), &dy_offset );
        err |= clSetKernelArg( kernel, arg++, sizeof(incy     ), &incy      );
        check_error( err );

        err = clEnqueueNDRangeKernel( queue, kernel, ndim, NULL, grid, threads, 0, NULL, NULL );
        check_error( err );
    }
}
