/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c

       auto-converted from zlange.cu
       @author Mark Gates
*/
#include "clmagma_runtime.h"
#include "common_magma.h"
#include "zlange.h"


/**
    Purpose
    -------
    ZLANGE  returns the value of the one norm, or the Frobenius norm, or
    the  infinity norm, or the  element of  largest absolute value  of a
    real matrix A.
    
    Description
    -----------
    ZLANGE returns the value
    
       ZLANGE = ( max(abs(A(i,j))), NORM = 'M' or 'm'
                (
                ( norm1(A),         NORM = '1', 'O' or 'o'
                (
                ( normI(A),         NORM = 'I' or 'i'
                (
                ( normF(A),         NORM = 'F', 'f', 'E' or 'e'  ** not yet supported
    
    where norm1 denotes the one norm of a matrix (maximum column sum),
    normI denotes the infinity norm of a matrix (maximum row sum) and
    normF denotes the Frobenius norm of a matrix (square root of sum of
    squares). Note that max(abs(A(i,j))) is not a consistent matrix norm.
    
    Arguments
    ---------
    @param[in]
    norm    CHARACTER*1
            Specifies the value to be returned in ZLANGE as described
            above.
    
    @param[in]
    m       INTEGER
            The number of rows of the matrix A.  M >= 0.  When M = 0,
            ZLANGE is set to zero.
    
    @param[in]
    n       INTEGER
            The number of columns of the matrix A.  N >= 0.  When N = 0,
            ZLANGE is set to zero.
    
    @param[in]
    dA      DOUBLE PRECISION array on the GPU, dimension (LDDA,N)
            The m by n matrix A.
    
    @param[in]
    ldda    INTEGER
            The leading dimension of the array A.  LDDA >= max(M,1).
    
    @param
    dwork   (workspace) DOUBLE PRECISION array on the GPU, dimension (LWORK).
    
    @param[in]
    lwork   INTEGER
            The dimension of the array WORK.
            If NORM = 'I' or 'M', LWORK >= max( 1, M ).
            If NORM = '1',        LWORK >= max( 1, N ).
            Note this is different than LAPACK, which requires WORK only for
            NORM = 'I', and does not pass LWORK.

    @ingroup magma_zaux2
    ********************************************************************/
extern "C" double
magmablas_zlange(
    magma_norm_t norm, magma_int_t m, magma_int_t n,
    magmaDoubleComplex_const_ptr dA, size_t dA_offset, magma_int_t ldda,
    magmaDouble_ptr dwork, size_t dwork_offset, magma_int_t lwork,
    magma_queue_t queue )
{
    cl_kernel kernel;
    cl_int err;
    int arg;

    magma_int_t info = 0;
    if ( ! (norm == MagmaInfNorm || norm == MagmaMaxNorm || norm == MagmaOneNorm) )
        info = -1;
    else if ( m < 0 )
        info = -2;
    else if ( n < 0 )
        info = -3;
    else if ( ldda < m )
        info = -5;
    else if ( ((norm == MagmaInfNorm || norm == MagmaMaxNorm) && (lwork < m)) ||
              ((norm == MagmaOneNorm) && (lwork < n)) )
        info = -7;

    if ( info != 0 ) {
        magma_xerbla( __func__, -(info) );
        return info;
    }
    
    /* Quick return */
    if ( m == 0 || n == 0 )
        return 0;
    
    const int ndim = 1;
    size_t threads[ndim];
    threads[0] = NB_X;
    double result = -1;
    if ( norm == MagmaInfNorm ) {
        size_t grid[ndim];
        grid[0] = magma_ceildiv( m, NB_X );
        grid[0] *= threads[0];
        kernel = g_runtime.get_kernel( "zlange_inf_kernel" );
        if ( kernel != NULL ) {
            err = 0;
            arg = 0;
            err |= clSetKernelArg( kernel, arg++, sizeof(m           ), &m            );
            err |= clSetKernelArg( kernel, arg++, sizeof(n           ), &n            );
            err |= clSetKernelArg( kernel, arg++, sizeof(dA          ), &dA           );
            err |= clSetKernelArg( kernel, arg++, sizeof(dA_offset   ), &dA_offset    );
            err |= clSetKernelArg( kernel, arg++, sizeof(ldda        ), &ldda         );
            err |= clSetKernelArg( kernel, arg++, sizeof(dwork       ), &dwork        );
            err |= clSetKernelArg( kernel, arg++, sizeof(dwork_offset), &dwork_offset );
            check_error( err );

            err = clEnqueueNDRangeKernel( queue, kernel, ndim, NULL, grid, threads, 0, NULL, NULL );
            check_error( err );
        }
        result = magmablas_dmax_nan( m, dwork, dwork_offset, queue );
    }
    else if ( norm == MagmaMaxNorm ) {
        size_t grid[ndim];
        grid[0] = magma_ceildiv( m, NB_X );
        grid[0] *= threads[0];
        kernel = g_runtime.get_kernel( "zlange_max_kernel" );
        if ( kernel != NULL ) {
            err = 0;
            arg = 0;
            err |= clSetKernelArg( kernel, arg++, sizeof(m           ), &m            );
            err |= clSetKernelArg( kernel, arg++, sizeof(n           ), &n            );
            err |= clSetKernelArg( kernel, arg++, sizeof(dA          ), &dA           );
            err |= clSetKernelArg( kernel, arg++, sizeof(dA_offset   ), &dA_offset    );
            err |= clSetKernelArg( kernel, arg++, sizeof(ldda        ), &ldda         );
            err |= clSetKernelArg( kernel, arg++, sizeof(dwork       ), &dwork        );
            err |= clSetKernelArg( kernel, arg++, sizeof(dwork_offset), &dwork_offset );
            check_error( err );

            err = clEnqueueNDRangeKernel( queue, kernel, ndim, NULL, grid, threads, 0, NULL, NULL );
            check_error( err );
        }
        result = magmablas_dmax_nan( m, dwork, dwork_offset, queue );
    }
    else if ( norm == MagmaOneNorm ) {
        size_t grid[ndim];
        grid[0] = n;
        grid[0] *= threads[0];
        kernel = g_runtime.get_kernel( "zlange_one_kernel" );
        if ( kernel != NULL ) {
            err = 0;
            arg = 0;
            err |= clSetKernelArg( kernel, arg++, sizeof(m           ), &m            );
            err |= clSetKernelArg( kernel, arg++, sizeof(n           ), &n            );
            err |= clSetKernelArg( kernel, arg++, sizeof(dA          ), &dA           );
            err |= clSetKernelArg( kernel, arg++, sizeof(dA_offset   ), &dA_offset    );
            err |= clSetKernelArg( kernel, arg++, sizeof(ldda        ), &ldda         );
            err |= clSetKernelArg( kernel, arg++, sizeof(dwork       ), &dwork        );
            err |= clSetKernelArg( kernel, arg++, sizeof(dwork_offset), &dwork_offset );
            check_error( err );

            err = clEnqueueNDRangeKernel( queue, kernel, ndim, NULL, grid, threads, 0, NULL, NULL );
            check_error( err );
        }
        result = magmablas_dmax_nan( n, dwork, dwork_offset, queue );  // note N instead of M
    }
    
    return result;
}
