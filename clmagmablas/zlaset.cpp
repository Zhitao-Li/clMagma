/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Mark Gates
       @author Azzam Haidar
       
       @precisions normal z -> s d c

       auto-converted from zlaset.cu

*/
#include "clmagma_runtime.h"
#include "common_magma.h"
#include "zlaset.h"

// To deal with really large matrices, this launchs multiple super blocks,
// each with up to 64K-1 x 64K-1 thread blocks, which is up to 4194240 x 4194240 matrix with BLK=64.
// CUDA architecture 2.0 limits each grid dimension to 64K-1.
// Instances arose for vectors used by sparse matrices with M > 4194240, though N is small.
const magma_int_t max_blocks = 65535;


//////////////////////////////////////////////////////////////////////////////////////
/**
    Purpose
    -------
    ZLASET initializes a 2-D array A to DIAG on the diagonal and
    OFFDIAG on the off-diagonals.
    
    Arguments
    ---------
    @param[in]
    uplo    magma_uplo_t
            Specifies the part of the matrix dA to be set.
      -     = MagmaUpper:      Upper triangular part
      -     = MagmaLower:      Lower triangular part
      -     = MagmaFull:       All of the matrix dA
    
    @param[in]
    m       INTEGER
            The number of rows of the matrix dA.  M >= 0.
    
    @param[in]
    n       INTEGER
            The number of columns of the matrix dA.  N >= 0.
    
    @param[in]
    offdiag COMPLEX_16
            The scalar OFFDIAG. (In LAPACK this is called ALPHA.)
    
    @param[in]
    diag    COMPLEX_16
            The scalar DIAG. (In LAPACK this is called BETA.)
    
    @param[in]
    dA      COMPLEX_16 array, dimension (LDDA,N)
            The M-by-N matrix dA.
            If UPLO = MagmaUpper, only the upper triangle or trapezoid is accessed;
            if UPLO = MagmaLower, only the lower triangle or trapezoid is accessed.
            On exit, A(i,j) = OFFDIAG, 1 <= i <= m, 1 <= j <= n, i != j;
            and      A(i,i) = DIAG,    1 <= i <= min(m,n)
    
    @param[in]
    ldda    INTEGER
            The leading dimension of the array dA.  LDDA >= max(1,M).
    
    @param[in]
    queue   magma_queue_t
            Queue to execute in.
    
    @ingroup magma_zaux2
    ********************************************************************/
extern "C"
void magmablas_zlaset(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magmaDoubleComplex offdiag, magmaDoubleComplex diag,
    magmaDoubleComplex_ptr dA, size_t dA_offset, magma_int_t ldda,
    magma_queue_t queue)
{
    cl_kernel kernel;
    cl_int err;
    int arg;

    magma_int_t info = 0;
    if ( uplo != MagmaLower && uplo != MagmaUpper && uplo != MagmaFull )
        info = -1;
    else if ( m < 0 )
        info = -2;
    else if ( n < 0 )
        info = -3;
    else if ( ldda < max(1,m) )
        info = -7;
    
    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;  //info;
    }
    
    if ( m == 0 || n == 0 ) {
        return;
    }
    
    assert( BLK_X == BLK_Y );
    const magma_int_t super_NB = max_blocks*BLK_X;
    const int ndim = 2;
    size_t super_grid[ndim];
    super_grid[0] = magma_ceildiv( m, super_NB );
    super_grid[1] = magma_ceildiv( n, super_NB );
    size_t dA_offset_ij;
    
    size_t threads[ndim];
    threads[0] = BLK_X;
    threads[1] = 1;
    size_t grid[ndim];
    
    magma_int_t mm, nn;
    if (uplo == MagmaLower) {
        for( unsigned int i=0; i < super_grid[0]; ++i ) {
            mm = (i == super_grid[0]-1 ? m % super_NB : super_NB);
            grid[0] = magma_ceildiv( mm, BLK_X );
            grid[0] *= threads[0];
            for( unsigned int j=0; j < super_grid[1] && j <= i; ++j ) {  // from left to diagonal
                nn = (j == super_grid[1]-1 ? n % super_NB : super_NB);
                grid[1] = magma_ceildiv( nn, BLK_Y );
                grid[1] *= threads[1];
                if ( i == j ) {  // diagonal super block
                    kernel = g_runtime.get_kernel( "zlaset_lower_kernel" );
                    if ( kernel != NULL ) {
                        err = 0;
                        arg = 0;
                        dA_offset_ij = dA_offset + i*super_NB + j*super_NB*ldda;
                        err |= clSetKernelArg( kernel, arg++, sizeof(mm          ), &mm           );
                        err |= clSetKernelArg( kernel, arg++, sizeof(nn          ), &nn           );
                        err |= clSetKernelArg( kernel, arg++, sizeof(offdiag     ), &offdiag      );
                        err |= clSetKernelArg( kernel, arg++, sizeof(diag        ), &diag         );
                        err |= clSetKernelArg( kernel, arg++, sizeof(dA          ), &dA           );
                        err |= clSetKernelArg( kernel, arg++, sizeof(dA_offset_ij), &dA_offset_ij );
                        err |= clSetKernelArg( kernel, arg++, sizeof(ldda        ), &ldda         );
                        check_error( err );

                        err = clEnqueueNDRangeKernel( queue, kernel, ndim, NULL, grid, threads, 0, NULL, NULL );
                        check_error( err );
                    }
                }
                else {           // off diagonal super block
                    kernel = g_runtime.get_kernel( "zlaset_full_kernel" );
                    if ( kernel != NULL ) {
                        err = 0;
                        arg = 0;
                        dA_offset_ij = dA_offset + i*super_NB + j*super_NB*ldda;
                        err |= clSetKernelArg( kernel, arg++, sizeof(mm          ), &mm           );
                        err |= clSetKernelArg( kernel, arg++, sizeof(nn          ), &nn           );
                        err |= clSetKernelArg( kernel, arg++, sizeof(offdiag     ), &offdiag      );
                        err |= clSetKernelArg( kernel, arg++, sizeof(diag        ), &diag         );
                        err |= clSetKernelArg( kernel, arg++, sizeof(dA          ), &dA           );
                        err |= clSetKernelArg( kernel, arg++, sizeof(dA_offset_ij), &dA_offset_ij );
                        err |= clSetKernelArg( kernel, arg++, sizeof(ldda        ), &ldda         );
                        check_error( err );

                        err = clEnqueueNDRangeKernel( queue, kernel, ndim, NULL, grid, threads, 0, NULL, NULL );
                        check_error( err );
                    }
                }
            }
        }
    }
    else if (uplo == MagmaUpper) {
        for( unsigned int i=0; i < super_grid[0]; ++i ) {
            mm = (i == super_grid[0]-1 ? m % super_NB : super_NB);
            grid[0] = magma_ceildiv( mm, BLK_X );
            grid[0] *= threads[0];
            for( unsigned int j=i; j < super_grid[1]; ++j ) {  // from diagonal to right
                nn = (j == super_grid[1]-1 ? n % super_NB : super_NB);
                grid[1] = magma_ceildiv( nn, BLK_Y );
                grid[1] *= threads[1];
                if ( i == j ) {  // diagonal super block
                    kernel = g_runtime.get_kernel( "zlaset_upper_kernel" );
                    if ( kernel != NULL ) {
                        err = 0;
                        arg = 0;
                        dA_offset_ij = dA_offset + i*super_NB + j*super_NB*ldda;
                        err |= clSetKernelArg( kernel, arg++, sizeof(mm          ), &mm           );
                        err |= clSetKernelArg( kernel, arg++, sizeof(nn          ), &nn           );
                        err |= clSetKernelArg( kernel, arg++, sizeof(offdiag     ), &offdiag      );
                        err |= clSetKernelArg( kernel, arg++, sizeof(diag        ), &diag         );
                        err |= clSetKernelArg( kernel, arg++, sizeof(dA          ), &dA           );
                        err |= clSetKernelArg( kernel, arg++, sizeof(dA_offset_ij), &dA_offset_ij );
                        err |= clSetKernelArg( kernel, arg++, sizeof(ldda        ), &ldda         );
                        check_error( err );

                        err = clEnqueueNDRangeKernel( queue, kernel, ndim, NULL, grid, threads, 0, NULL, NULL );
                        check_error( err );
                    }
                }
                else {           // off diagonal super block
                    kernel = g_runtime.get_kernel( "zlaset_full_kernel" );
                    if ( kernel != NULL ) {
                        err = 0;
                        arg = 0;
                        dA_offset_ij = dA_offset + i*super_NB + j*super_NB*ldda;
                        err |= clSetKernelArg( kernel, arg++, sizeof(mm          ), &mm           );
                        err |= clSetKernelArg( kernel, arg++, sizeof(nn          ), &nn           );
                        err |= clSetKernelArg( kernel, arg++, sizeof(offdiag     ), &offdiag      );
                        err |= clSetKernelArg( kernel, arg++, sizeof(diag        ), &diag         );
                        err |= clSetKernelArg( kernel, arg++, sizeof(dA          ), &dA           );
                        err |= clSetKernelArg( kernel, arg++, sizeof(dA_offset_ij), &dA_offset_ij );
                        err |= clSetKernelArg( kernel, arg++, sizeof(ldda        ), &ldda         );
                        check_error( err );

                        err = clEnqueueNDRangeKernel( queue, kernel, ndim, NULL, grid, threads, 0, NULL, NULL );
                        check_error( err );
                    }
                }
            }
        }
    }
    else {
        for( unsigned int i=0; i < super_grid[0]; ++i ) {
            mm = (i == super_grid[0]-1 ? m % super_NB : super_NB);
            grid[0] = magma_ceildiv( mm, BLK_X );
            grid[0] *= threads[0];
            for( unsigned int j=0; j < super_grid[1]; ++j ) {  // full row
                nn = (j == super_grid[1]-1 ? n % super_NB : super_NB);
                grid[1] = magma_ceildiv( nn, BLK_Y );
                grid[1] *= threads[1];
                if ( i == j ) {  // diagonal super block
                    kernel = g_runtime.get_kernel( "zlaset_full_kernel" );
                    if ( kernel != NULL ) {
                        err = 0;
                        arg = 0;
                        dA_offset_ij = dA_offset + i*super_NB + j*super_NB*ldda;
                        err |= clSetKernelArg( kernel, arg++, sizeof(mm          ), &mm           );
                        err |= clSetKernelArg( kernel, arg++, sizeof(nn          ), &nn           );
                        err |= clSetKernelArg( kernel, arg++, sizeof(offdiag     ), &offdiag      );
                        err |= clSetKernelArg( kernel, arg++, sizeof(diag        ), &diag         );
                        err |= clSetKernelArg( kernel, arg++, sizeof(dA          ), &dA           );
                        err |= clSetKernelArg( kernel, arg++, sizeof(dA_offset_ij), &dA_offset_ij );
                        err |= clSetKernelArg( kernel, arg++, sizeof(ldda        ), &ldda         );
                        check_error( err );

                        err = clEnqueueNDRangeKernel( queue, kernel, ndim, NULL, grid, threads, 0, NULL, NULL );
                        check_error( err );
                    }
                }
                else {           // off diagonal super block
                    kernel = g_runtime.get_kernel( "zlaset_full_kernel" );
                    if ( kernel != NULL ) {
                        err = 0;
                        i   = 0;
                        dA_offset_ij = dA_offset + i*super_NB + j*super_NB*ldda;
                        err |= clSetKernelArg( kernel, arg++, sizeof(mm          ), &mm           );
                        err |= clSetKernelArg( kernel, arg++, sizeof(nn          ), &nn           );
                        err |= clSetKernelArg( kernel, arg++, sizeof(offdiag     ), &offdiag      );
                        err |= clSetKernelArg( kernel, arg++, sizeof(diag        ), &diag         );
                        err |= clSetKernelArg( kernel, arg++, sizeof(dA          ), &dA           );
                        err |= clSetKernelArg( kernel, arg++, sizeof(dA_offset_ij), &dA_offset_ij );
                        err |= clSetKernelArg( kernel, arg++, sizeof(ldda        ), &ldda         );
                        check_error( err );

                        err = clEnqueueNDRangeKernel( queue, kernel, ndim, NULL, grid, threads, 0, NULL, NULL );
                        check_error( err );
                    }
                }
            }
        }
    }
}
