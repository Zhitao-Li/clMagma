/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @generated from clmagmablas/zsymmetrize.cpp, normal z -> c, Tue Jun 18 16:14:18 2019

       auto-converted from csymmetrize.cu
       @author Mark Gates
*/
#include "clmagma_runtime.h"
#include "common_magma.h"
#include "csymmetrize.h"


/**
    Purpose
    -------
    
    CSYMMETRIZE copies lower triangle to upper triangle, or vice-versa,
    to make dA a general representation of a symmetric matrix.
    
    Arguments
    ---------
    
    @param[in]
    uplo    magma_uplo_t
            Specifies the part of the matrix dA that is valid on input.
      -     = MagmaUpper:      Upper triangular part
      -     = MagmaLower:      Lower triangular part
    
    @param[in]
    m       INTEGER
            The number of rows of the matrix dA.  M >= 0.
    
    @param[in,out]
    dA      COMPLEX array, dimension (LDDA,N)
            The m by m matrix dA.
    
    @param[in]
    ldda    INTEGER
            The leading dimension of the array dA.  LDDA >= max(1,M).
    
    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_caux2
    ********************************************************************/
extern "C" void
magmablas_csymmetrize(
    magma_uplo_t uplo, magma_int_t m,
    magmaFloatComplex_ptr dA, size_t dA_offset, magma_int_t ldda,
    magma_queue_t queue )
{
    cl_kernel kernel;
    cl_int err;
    int arg;

    magma_int_t info = 0;
    if ( uplo != MagmaLower && uplo != MagmaUpper )
        info = -1;
    else if ( m < 0 )
        info = -2;
    else if ( ldda < max(1,m) )
        info = -4;
    
    if ( info != 0 ) {
        magma_xerbla( __func__, -(info) );
        return;
    }
    
    if ( m == 0 )
        return;
    
    const int ndim = 1;
    size_t threads[ndim];
    threads[0] = NB;
    size_t grid[ndim];
    grid[0] = magma_ceildiv( m, NB );
    grid[0] *= threads[0];
    
    if ( uplo == MagmaUpper ) {
        kernel = g_runtime.get_kernel( "csymmetrize_upper" );
        if ( kernel != NULL ) {
            err = 0;
            arg = 0;
            err |= clSetKernelArg( kernel, arg++, sizeof(m        ), &m         );
            err |= clSetKernelArg( kernel, arg++, sizeof(dA       ), &dA        );
            err |= clSetKernelArg( kernel, arg++, sizeof(dA_offset), &dA_offset );
            err |= clSetKernelArg( kernel, arg++, sizeof(ldda     ), &ldda      );
            check_error( err );

            err = clEnqueueNDRangeKernel( queue, kernel, ndim, NULL, grid, threads, 0, NULL, NULL );
            check_error( err );
        }
    }
    else {
        kernel = g_runtime.get_kernel( "csymmetrize_lower" );
        if ( kernel != NULL ) {
            err = 0;
            arg = 0;
            err |= clSetKernelArg( kernel, arg++, sizeof(m        ), &m         );
            err |= clSetKernelArg( kernel, arg++, sizeof(dA       ), &dA        );
            err |= clSetKernelArg( kernel, arg++, sizeof(dA_offset), &dA_offset );
            err |= clSetKernelArg( kernel, arg++, sizeof(ldda     ), &ldda      );
            check_error( err );

            err = clEnqueueNDRangeKernel( queue, kernel, ndim, NULL, grid, threads, 0, NULL, NULL );
            check_error( err );
        }
    }
}
