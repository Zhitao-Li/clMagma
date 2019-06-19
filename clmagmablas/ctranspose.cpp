/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @generated from clmagmablas/ztranspose.cpp, normal z -> c, Tue Jun 18 16:14:18 2019

       auto-converted from ctranspose.cu

       @author Stan Tomov
       @author Mark Gates
*/
#include "clmagma_runtime.h"
#include "common_magma.h"
#include "ctranspose.h"


/**
    Purpose
    -------
    ctranspose_q copies and transposes a matrix dA to matrix dAT.
    
    Same as ctranspose, but adds queue argument.
        
    Arguments
    ---------
    @param[in]
    m       INTEGER
            The number of rows of the matrix dA.  M >= 0.
    
    @param[in]
    n       INTEGER
            The number of columns of the matrix dA.  N >= 0.
    
    @param[in]
    dA      COMPLEX array, dimension (LDDA,N)
            The M-by-N matrix dA.
    
    @param[in]
    ldda    INTEGER
            The leading dimension of the array dA.  LDDA >= M.
    
    @param[in]
    dAT     COMPLEX array, dimension (LDDAT,M)
            The N-by-M matrix dAT.
    
    @param[in]
    lddat   INTEGER
            The leading dimension of the array dAT.  LDDAT >= N.
    
    @param[in]
    queue   magma_queue_t
            Queue to execute in.
    
    @ingroup magma_caux2
    ********************************************************************/
extern "C" void
magmablas_ctranspose(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex_const_ptr dA, size_t dA_offset,  magma_int_t ldda,
    magmaFloatComplex_ptr       dAT, size_t dAT_offset, magma_int_t lddat,
    magma_queue_t queue )
{
    cl_kernel kernel;
    cl_int err;
    int arg;

    magma_int_t info = 0;
    if ( m < 0 )
        info = -1;
    else if ( n < 0 )
        info = -2;
    else if ( ldda < m )
        info = -4;
    else if ( lddat < n )
        info = -6;
    
    if ( info != 0 ) {
        magma_xerbla( __func__, -(info) );
        return;  //info;
    }
    
    /* Quick return */
    if ( (m == 0) || (n == 0) )
        return;

    const int ndim = 2;
    size_t threads[ndim];
    threads[0] = NX;
    threads[1] = NY;
    size_t grid[ndim];
    grid[0] = magma_ceildiv( m, NB );
    grid[1] = magma_ceildiv( n, NB );
    grid[0] *= threads[0];
    grid[1] *= threads[1];
    kernel = g_runtime.get_kernel( "ctranspose_kernel" );
    if ( kernel != NULL ) {
        err = 0;
        arg = 0;
        err |= clSetKernelArg( kernel, arg++, sizeof(m         ), &m          );
        err |= clSetKernelArg( kernel, arg++, sizeof(n         ), &n          );
        err |= clSetKernelArg( kernel, arg++, sizeof(dA        ), &dA         );
        err |= clSetKernelArg( kernel, arg++, sizeof(dA_offset ), &dA_offset  );
        err |= clSetKernelArg( kernel, arg++, sizeof(ldda      ), &ldda       );
        err |= clSetKernelArg( kernel, arg++, sizeof(dAT       ), &dAT        );
        err |= clSetKernelArg( kernel, arg++, sizeof(dAT_offset), &dAT_offset );
        err |= clSetKernelArg( kernel, arg++, sizeof(lddat     ), &lddat      );
        check_error( err );

        err = clEnqueueNDRangeKernel( queue, kernel, ndim, NULL, grid, threads, 0, NULL, NULL );
        check_error( err );
    }
}
