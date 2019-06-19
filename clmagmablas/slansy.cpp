/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Mark Gates
       @generated from clmagmablas/zlanhe.cpp, normal z -> s, Tue Jun 18 16:14:18 2019

       auto-converted from slansy.cu

*/
#include "clmagma_runtime.h"
#include "common_magma.h"
#include "slansy.h"


/* Computes row sums dwork[i] = sum( abs( A(i,:) )), i=0:n-1, for || A ||_inf */
extern "C" void
slansy_inf(
    magma_uplo_t uplo, magma_int_t n,
    magmaFloat_const_ptr A, size_t A_offset, magma_int_t lda,
    magmaFloat_ptr dwork, size_t dwork_offset,
    magma_queue_t queue )
{
    cl_kernel kernel;
    cl_int err;
    int arg;

    const int ndim = 2;
    size_t threads[ndim];
    threads[0] = inf_bs;
    threads[1] = 4;
    size_t grid[ndim];
    grid[0] = magma_ceildiv( n, inf_bs );
    grid[1] = 1;
    grid[0] *= threads[0];
    grid[1] *= threads[1];

    int n_full_block = (n - n % inf_bs) /inf_bs;
    int n_mod_bs = n % inf_bs;
    if ( uplo == MagmaLower) {
        kernel = g_runtime.get_kernel( "slansy_inf_kernel_lower" );
        if ( kernel != NULL ) {
            err = 0;
            arg = 0;
            err |= clSetKernelArg( kernel, arg++, sizeof(n           ), &n            );
            err |= clSetKernelArg( kernel, arg++, sizeof(A           ), &A            );
            err |= clSetKernelArg( kernel, arg++, sizeof(A_offset    ), &A_offset     );
            err |= clSetKernelArg( kernel, arg++, sizeof(lda         ), &lda          );
            err |= clSetKernelArg( kernel, arg++, sizeof(dwork       ), &dwork        );
            err |= clSetKernelArg( kernel, arg++, sizeof(dwork_offset), &dwork_offset );
            err |= clSetKernelArg( kernel, arg++, sizeof(n_full_block), &n_full_block );
            err |= clSetKernelArg( kernel, arg++, sizeof(n_mod_bs    ), &n_mod_bs     );
            check_error( err );

            err = clEnqueueNDRangeKernel( queue, kernel, ndim, NULL, grid, threads, 0, NULL, NULL );
            check_error( err );
        }
    }
    else {
        kernel = g_runtime.get_kernel( "slansy_inf_kernel_upper" );
        if ( kernel != NULL ) {
            err = 0;
            arg = 0;
            err |= clSetKernelArg( kernel, arg++, sizeof(n           ), &n            );
            err |= clSetKernelArg( kernel, arg++, sizeof(A           ), &A            );
            err |= clSetKernelArg( kernel, arg++, sizeof(A_offset    ), &A_offset     );
            err |= clSetKernelArg( kernel, arg++, sizeof(lda         ), &lda          );
            err |= clSetKernelArg( kernel, arg++, sizeof(dwork       ), &dwork        );
            err |= clSetKernelArg( kernel, arg++, sizeof(dwork_offset), &dwork_offset );
            err |= clSetKernelArg( kernel, arg++, sizeof(n_full_block), &n_full_block );
            err |= clSetKernelArg( kernel, arg++, sizeof(n_mod_bs    ), &n_mod_bs     );
            check_error( err );

            err = clEnqueueNDRangeKernel( queue, kernel, ndim, NULL, grid, threads, 0, NULL, NULL );
            check_error( err );
        }
    }
}


/* Computes dwork[i] = max( abs( A(i,:) )), i=0:n-1, for ||A||_max */
extern "C" void
slansy_max(
    magma_uplo_t uplo, magma_int_t n,
    magmaFloat_const_ptr A, size_t A_offset, magma_int_t lda,
    magmaFloat_ptr dwork, size_t dwork_offset,
    magma_queue_t queue )
{
    cl_kernel kernel;
    cl_int err;
    int arg;

    const int ndim = 1;
    size_t threads[ndim];
    threads[0] = max_bs;
    size_t grid[ndim];
    grid[0] = magma_ceildiv( n, max_bs );
    grid[0] *= threads[0];

    if ( uplo == MagmaLower ) {
        kernel = g_runtime.get_kernel( "slansy_max_kernel_lower" );
        if ( kernel != NULL ) {
            err = 0;
            arg = 0;
            err |= clSetKernelArg( kernel, arg++, sizeof(n           ), &n            );
            err |= clSetKernelArg( kernel, arg++, sizeof(A           ), &A            );
            err |= clSetKernelArg( kernel, arg++, sizeof(A_offset    ), &A_offset     );
            err |= clSetKernelArg( kernel, arg++, sizeof(lda         ), &lda          );
            err |= clSetKernelArg( kernel, arg++, sizeof(dwork       ), &dwork        );
            err |= clSetKernelArg( kernel, arg++, sizeof(dwork_offset), &dwork_offset );
            check_error( err );

            err = clEnqueueNDRangeKernel( queue, kernel, ndim, NULL, grid, threads, 0, NULL, NULL );
            check_error( err );
        }
    }
    else {
        kernel = g_runtime.get_kernel( "slansy_max_kernel_upper" );
        if ( kernel != NULL ) {
            err = 0;
            arg = 0;
            err |= clSetKernelArg( kernel, arg++, sizeof(n           ), &n            );
            err |= clSetKernelArg( kernel, arg++, sizeof(A           ), &A            );
            err |= clSetKernelArg( kernel, arg++, sizeof(A_offset    ), &A_offset     );
            err |= clSetKernelArg( kernel, arg++, sizeof(lda         ), &lda          );
            err |= clSetKernelArg( kernel, arg++, sizeof(dwork       ), &dwork        );
            err |= clSetKernelArg( kernel, arg++, sizeof(dwork_offset), &dwork_offset );
            check_error( err );

            err = clEnqueueNDRangeKernel( queue, kernel, ndim, NULL, grid, threads, 0, NULL, NULL );
            check_error( err );
        }
    }
}


/* ====================================================================== */
/**
    Purpose
    -------
    SLANSY returns the value of the one norm, or the Frobenius norm, or
    the infinity norm, or the element of largest absolute value of a
    real symmetric matrix A.
    
       SLANSY = ( max(abs(A(i,j))), NORM = 'M' or 'm'
                (
                ( norm1(A),         NORM = '1', 'O' or 'o'      ** supported only for (PRECISION_s || PRECISION_d || PRECISION_c || __CUDA_ARCH__ >= 200)
                (
                ( normI(A),         NORM = 'I' or 'i'           ** supported only for (PRECISION_s || PRECISION_d || PRECISION_c || __CUDA_ARCH__ >= 200)
                (
                ( normF(A),         NORM = 'F', 'f', 'E' or 'e' ** not yet supported
    
    where norm1 denotes the one norm of a matrix (maximum column sum),
    normI denotes the infinity norm of a matrix (maximum row sum) and
    normF denotes the Frobenius norm of a matrix (square root of sum of squares).
    Note that max(abs(A(i,j))) is not a consistent matrix norm.
    
    On error, returns SLANSY < 0: if SLANSY = -i, the i-th argument had an illegal value.
    
    Arguments:
    ----------
    @param[in]
    norm    CHARACTER*1
            Specifies the value to be returned in SLANSY as described above.
    
    @param[in]
    uplo    magma_uplo_t
            Specifies whether the upper or lower triangular part of the
            symmetric matrix A is to be referenced.
      -     = MagmaUpper: Upper triangular part of A is referenced
      -     = MagmaLower: Lower triangular part of A is referenced
    
    @param[in]
    n       INTEGER
            The order of the matrix A. N >= 0. When N = 0, SLANSY is
            set to zero.
    
    @param[in]
    dA      REAL array on the GPU, dimension (LDDA,N)
            The symmetric matrix A. If UPLO = MagmaUpper, the leading n by n
            upper triangular part of A contains the upper triangular part
            of the matrix A, and the strictly lower triangular part of A
            is not referenced. If UPLO = MagmaLower, the leading n by n lower
            triangular part of A contains the lower triangular part of
            the matrix A, and the strictly upper triangular part of A is
            not referenced. Note that the imaginary parts of the diagonal
            elements need not be set and are assumed to be zero.
    
    @param[in]
    ldda    INTEGER
            The leading dimension of the array A. LDDA >= max(N,1).
    
    @param
    dwork   (workspace) REAL array on the GPU, dimension (MAX(1,LWORK)),
            where LWORK >= N.
            NOTE: this is different than LAPACK, where WORK is required
            only for norm1 and normI. Here max-norm also requires work.
    
    @ingroup magma_saux2
    ********************************************************************/
extern "C" float
magmablas_slansy(
    magma_norm_t norm, magma_uplo_t uplo, magma_int_t n,
    magmaFloat_const_ptr dA, size_t dA_offset, magma_int_t ldda,
    magmaFloat_ptr dwork, size_t dwork_offset, magma_int_t lwork,
    magma_queue_t queue )
{
    magma_int_t info = 0;

    // 1-norm == inf-norm since A is symmetric
    bool inf_norm = (norm == MagmaInfNorm || norm == MagmaOneNorm);
    bool max_norm = (norm == MagmaMaxNorm);
    
    // inf_norm Double-Complex requires > 16 KB shared data (arch >= 200)
    const bool inf_implemented = true;
    
    if ( ! (max_norm || (inf_norm && inf_implemented)) )
        info = -1;
    else if ( uplo != MagmaUpper && uplo != MagmaLower )
        info = -2;
    else if ( n < 0 )
        info = -3;
    else if ( ldda < n )
        info = -5;
    else if ( lwork < n )
        info = -7;
    
    if ( info != 0 ) {
        magma_xerbla( __func__, -(info) );
        return info;
    }
    
    /* Quick return */
    if ( n == 0 )
        return 0;
        
    if ( inf_norm ) {
        slansy_inf( uplo, n, dA, dA_offset, ldda, dwork, dwork_offset, queue );
    }
    else {
        slansy_max( uplo, n, dA, dA_offset, ldda, dwork, dwork_offset, queue );
    }
    float res = magmablas_smax_nan( n, dwork, dwork_offset, queue );
    return res;
}
