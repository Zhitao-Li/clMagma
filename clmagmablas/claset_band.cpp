/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Raffaele Solca
       @author Mark Gates
       
       @generated from clmagmablas/zlaset_band.cpp, normal z -> c, Tue Jun 18 16:14:18 2019

       auto-converted from claset_band.cu

*/
#include "clmagma_runtime.h"
#include "common_magma.h"
#include "claset_band.h"


/**
    Purpose
    -------
    CLASET_BAND initializes the main diagonal of dA to DIAG,
    and the K-1 sub- or super-diagonals to OFFDIAG.
    
    Arguments
    ---------
    
    @param[in]
    uplo    magma_uplo_t
            Specifies the part of the matrix dA to be set.
      -     = MagmaUpper:      Upper triangular part
      -     = MagmaLower:      Lower triangular part
    
    @param[in]
    m       INTEGER
            The number of rows of the matrix dA.  M >= 0.
    
    @param[in]
    n       INTEGER
            The number of columns of the matrix dA.  N >= 0.
    
    @param[in]
    k       INTEGER
            The number of diagonals to set, including the main diagonal.  K >= 0.
            Currently, K <= 1024 due to CUDA restrictions (max. number of threads per block).
    
    @param[in]
    offdiag COMPLEX
            Off-diagonal elements in the band are set to OFFDIAG.
    
    @param[in]
    diag    COMPLEX
            All the main diagonal elements are set to DIAG.
    
    @param[in]
    dA      COMPLEX array, dimension (LDDA,N)
            The M-by-N matrix dA.
            If UPLO = MagmaUpper, only the upper triangle or trapezoid is accessed;
            if UPLO = MagmaLower, only the lower triangle or trapezoid is accessed.
            On exit, A(i,j) = ALPHA, 1 <= i <= m, 1 <= j <= n where i != j, abs(i-j) < k;
            and      A(i,i) = BETA,  1 <= i <= min(m,n)
    
    @param[in]
    ldda    INTEGER
            The leading dimension of the array dA.  LDDA >= max(1,M).
    
    @param[in]
    queue   magma_queue_t
            Stream to execute CLASET in.
    
    @ingroup magma_caux2
    ********************************************************************/
extern "C" void
magmablas_claset_band(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n, magma_int_t k,
    magmaFloatComplex offdiag, magmaFloatComplex diag,
    magmaFloatComplex_ptr dA, size_t dA_offset, magma_int_t ldda,
    magma_queue_t queue)
{
    cl_kernel kernel;
    cl_int err;
    int arg;

    magma_int_t info = 0;
    if ( uplo != MagmaLower && uplo != MagmaUpper )
        info = -1;
    else if ( m < 0 )
        info = -2;
    else if ( n < 0 )
        info = -3;
    else if ( k < 0 || k > 1024 )
        info = -4;
    else if ( ldda < max(1,m) )
        info = -6;
    
    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;  //info;
    }
    
    if (uplo == MagmaUpper) {
        const int ndim = 1;
        size_t threads[ndim];
        threads[0] = min(k,n);
        size_t grid[ndim];
        grid[0] = magma_ceildiv( min(m+k-1,n), NB );
        grid[0] *= threads[0];
        kernel = g_runtime.get_kernel( "claset_band_upper" );
        if ( kernel != NULL ) {
            err = 0;
            arg = 0;
            err |= clSetKernelArg( kernel, arg++, sizeof(m        ), &m         );
            err |= clSetKernelArg( kernel, arg++, sizeof(n        ), &n         );
            err |= clSetKernelArg( kernel, arg++, sizeof(offdiag  ), &offdiag   );
            err |= clSetKernelArg( kernel, arg++, sizeof(diag     ), &diag      );
            err |= clSetKernelArg( kernel, arg++, sizeof(dA       ), &dA        );
            err |= clSetKernelArg( kernel, arg++, sizeof(dA_offset), &dA_offset );
            err |= clSetKernelArg( kernel, arg++, sizeof(ldda     ), &ldda      );
            check_error( err );

            err = clEnqueueNDRangeKernel( queue, kernel, ndim, NULL, grid, threads, 0, NULL, NULL );
            check_error( err );
        }
    }
    else if (uplo == MagmaLower) {
        const int ndim = 1;
        size_t threads[ndim];
        threads[0] = min(k,m);
        size_t grid[ndim];
        grid[0] = magma_ceildiv( min(m,n), NB );
        grid[0] *= threads[0];
        kernel = g_runtime.get_kernel( "claset_band_lower" );
        if ( kernel != NULL ) {
            err = 0;
            arg = 0;
            err |= clSetKernelArg( kernel, arg++, sizeof(m        ), &m         );
            err |= clSetKernelArg( kernel, arg++, sizeof(n        ), &n         );
            err |= clSetKernelArg( kernel, arg++, sizeof(offdiag  ), &offdiag   );
            err |= clSetKernelArg( kernel, arg++, sizeof(diag     ), &diag      );
            err |= clSetKernelArg( kernel, arg++, sizeof(dA       ), &dA        );
            err |= clSetKernelArg( kernel, arg++, sizeof(dA_offset), &dA_offset );
            err |= clSetKernelArg( kernel, arg++, sizeof(ldda     ), &ldda      );
            check_error( err );

            err = clEnqueueNDRangeKernel( queue, kernel, ndim, NULL, grid, threads, 0, NULL, NULL );
            check_error( err );
        }
    }
}
