/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @generated from clmagmablas/zgeadd.cpp, normal z -> c, Tue Jun 18 16:14:18 2019

       auto-converted from cgeadd.cu
       @author Mark Gates
*/
#include "clmagma_runtime.h"
#include "common_magma.h"
#include "cgeadd.h"


/**
    Purpose
    -------
    ZGEADD adds two matrices, dB = alpha*dA + dB.
    
    Arguments
    ---------
    
    @param[in]
    m       INTEGER
            The number of rows of the matrix dA.  M >= 0.
    
    @param[in]
    n       INTEGER
            The number of columns of the matrix dA.  N >= 0.
    
    @param[in]
    alpha   COMPLEX
            The scalar alpha.
            
    @param[in]
    dA      COMPLEX array, dimension (LDDA,N)
            The m by n matrix dA.
    
    @param[in]
    ldda    INTEGER
            The leading dimension of the array dA.  LDDA >= max(1,M).
            
    @param[in,out]
    dB      COMPLEX array, dimension (LDDB,N)
            The m by n matrix dB.
    
    @param[in]
    lddb    INTEGER
            The leading dimension of the array dB.  LDDB >= max(1,M).
    
    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_caux2
    ********************************************************************/
extern "C" void
magmablas_cgeadd(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, size_t dA_offset, magma_int_t ldda,
    magmaFloatComplex_ptr       dB, size_t dB_offset, magma_int_t lddb,
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
    else if ( ldda < max(1,m))
        info = -5;
    else if ( lddb < max(1,m))
        info = -7;
    
    if ( info != 0 ) {
        magma_xerbla( __func__, -(info) );
        return;
    }
    
    if ( m == 0 || n == 0 )
        return;
    
    const int ndim = 2;
    size_t threads[ndim];
    threads[0] = BLK_X;
    threads[1] = 1;
    size_t grid[ndim];
    grid[0] = magma_ceildiv( m, BLK_X );
    grid[1] = magma_ceildiv( n, BLK_Y );
    grid[0] *= threads[0];
    grid[1] *= threads[1];
    
    kernel = g_runtime.get_kernel( "cgeadd_full" );
    if ( kernel != NULL ) {
        err = 0;
        arg = 0;
        err |= clSetKernelArg( kernel, arg++, sizeof(m        ), &m         );
        err |= clSetKernelArg( kernel, arg++, sizeof(n        ), &n         );
        err |= clSetKernelArg( kernel, arg++, sizeof(alpha    ), &alpha     );
        err |= clSetKernelArg( kernel, arg++, sizeof(dA       ), &dA        );
        err |= clSetKernelArg( kernel, arg++, sizeof(dA_offset), &dA_offset );
        err |= clSetKernelArg( kernel, arg++, sizeof(ldda     ), &ldda      );
        err |= clSetKernelArg( kernel, arg++, sizeof(dB       ), &dB        );
        err |= clSetKernelArg( kernel, arg++, sizeof(dB_offset), &dB_offset );
        err |= clSetKernelArg( kernel, arg++, sizeof(lddb     ), &lddb      );
        check_error( err );

        err = clEnqueueNDRangeKernel( queue, kernel, ndim, NULL, grid, threads, 0, NULL, NULL );
        check_error( err );
    }
}
