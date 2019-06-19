/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c

       auto-converted from ztranspose_inplace.cu

       @author Stan Tomov
       @author Mark Gates
*/
#include "clmagma_runtime.h"
#include "common_magma.h"
#include "ztranspose_inplace.h"


/**
    Purpose
    -------
    ztranspose_inplace_q transposes a square N-by-N matrix in-place.
    
    Same as ztranspose_inplace, but adds queue argument.
    
    Arguments
    ---------
    @param[in]
    n       INTEGER
            The number of rows & columns of the matrix dA.  N >= 0.
    
    @param[in]
    dA      COMPLEX_16 array, dimension (LDDA,N)
            The N-by-N matrix dA.
            On exit, dA(j,i) = dA_original(i,j), for 0 <= i,j < N.
    
    @param[in]
    ldda    INTEGER
            The leading dimension of the array dA.  LDDA >= N.
    
    @param[in]
    queue   magma_queue_t
            Queue to execute in.
    
    @ingroup magma_zaux2
    ********************************************************************/
extern "C" void
magmablas_ztranspose_inplace(
    magma_int_t n,
    magmaDoubleComplex_ptr dA, size_t dA_offset, magma_int_t ldda,
    magma_queue_t queue )
{
    cl_kernel kernel;
    cl_int err;
    int arg;

    magma_int_t info = 0;
    if ( n < 0 )
        info = -1;
    else if ( ldda < n )
        info = -3;
    
    if ( info != 0 ) {
        magma_xerbla( __func__, -(info) );
        return;  //info;
    }
    
    const int ndim = 2;
    size_t threads[ndim];
    threads[0] = NB;
    threads[1] = NB;
    int nblock = magma_ceildiv( n, NB );
    
    // need 1/2 * (nblock+1) * nblock to cover lower triangle and diagonal of matrix.
    // block assignment differs depending on whether nblock is odd or even.
    if ( nblock % 2 == 1 ) {
        size_t grid[ndim];
        grid[0] = nblock;
        grid[1] = (nblock+1)/2;
        grid[0] *= threads[0];
        grid[1] *= threads[1];
        kernel = g_runtime.get_kernel( "ztranspose_inplace_odd" );
        if ( kernel != NULL ) {
            err = 0;
            arg = 0;
            err |= clSetKernelArg( kernel, arg++, sizeof(n        ), &n         );
            err |= clSetKernelArg( kernel, arg++, sizeof(dA       ), &dA        );
            err |= clSetKernelArg( kernel, arg++, sizeof(dA_offset), &dA_offset );
            err |= clSetKernelArg( kernel, arg++, sizeof(ldda     ), &ldda      );
            check_error( err );

            err = clEnqueueNDRangeKernel( queue, kernel, ndim, NULL, grid, threads, 0, NULL, NULL );
            check_error( err );
        }
    }
    else {
        size_t grid[ndim];
        grid[0] = nblock+1;
        grid[1] = nblock/2;
        grid[0] *= threads[0];
        grid[1] *= threads[1];
        kernel = g_runtime.get_kernel( "ztranspose_inplace_even" );
        if ( kernel != NULL ) {
            err = 0;
            arg = 0;
            err |= clSetKernelArg( kernel, arg++, sizeof(n        ), &n         );
            err |= clSetKernelArg( kernel, arg++, sizeof(dA       ), &dA        );
            err |= clSetKernelArg( kernel, arg++, sizeof(dA_offset), &dA_offset );
            err |= clSetKernelArg( kernel, arg++, sizeof(ldda     ), &ldda      );
            check_error( err );

            err = clEnqueueNDRangeKernel( queue, kernel, ndim, NULL, grid, threads, 0, NULL, NULL );
            check_error( err );
        }
    }
}
