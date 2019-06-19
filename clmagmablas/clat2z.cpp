/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions mixed zc -> ds

       auto-converted from clat2z.cu
       @author Mark Gates
*/
#include "clmagma_runtime.h"
#include "common_magma.h"
#include "clat2z.h"


/**
    Purpose
    -------
    CLAT2Z converts a single-complex matrix, SA,
                 to a double-complex matrix, A.

    Note that while it is possible to overflow while converting
    from double to single, it is not possible to overflow when
    converting from single to double.

    Arguments
    ---------
    @param[in]
    uplo    magma_uplo_t
            Specifies the part of the matrix A to be converted.
      -     = MagmaUpper:      Upper triangular part
      -     = MagmaLower:      Lower triangular part
    
    @param[in]
    n       INTEGER
            The number of columns of the matrix A.  n >= 0.
    
    @param[in]
    A       COMPLEX_16 array, dimension (LDA,n)
            On entry, the n-by-n coefficient matrix A.
    
    @param[in]
    lda     INTEGER
            The leading dimension of the array A.  LDA >= max(1,n).
    
    @param[out]
    SA      COMPLEX array, dimension (LDSA,n)
            On exit, if INFO=0, the n-by-n coefficient matrix SA;
            if INFO > 0, the content of SA is unspecified.
    
    @param[in]
    ldsa    INTEGER
            The leading dimension of the array SA.  LDSA >= max(1,n).
    
    @param[out]
    info    INTEGER
      -     = 0:  successful exit.
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
    
    @param[in]
    queue   magma_queue_t
            Queue to execute in.
    
    @ingroup magma_zaux2
    ********************************************************************/
extern "C" void
magmablas_clat2z(
    magma_uplo_t uplo, magma_int_t n,
    magmaFloatComplex_const_ptr SA, size_t SA_offset, magma_int_t ldsa,
    magmaDoubleComplex_ptr      A, size_t A_offset,  magma_int_t lda,
    magma_queue_t queue,
    magma_int_t *info )
{
    cl_kernel kernel;
    cl_int err;
    int arg;

    *info = 0;
    if ( uplo != MagmaLower && uplo != MagmaUpper )
        *info = -1;
    else if ( n < 0 )
        *info = -2;
    else if ( lda < max(1,n) )
        *info = -4;
    else if ( ldsa < max(1,n) )
        *info = -6;
    
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return; //*info;
    }

    /* quick return */
    if ( n == 0 ) {
        return;
    }
    
    const int ndim = 2;
    size_t threads[ndim];
    threads[0] = BLK_X;
    threads[1] = 1;
    size_t grid[ndim];
    grid[0] = magma_ceildiv( n, BLK_X );
    grid[1] = magma_ceildiv( n, BLK_Y );
    grid[0] *= threads[0];
    grid[1] *= threads[1];
    
    if (uplo == MagmaLower) {
        kernel = g_runtime.get_kernel( "clat2z_lower" );
        if ( kernel != NULL ) {
            err = 0;
            arg = 0;
            err |= clSetKernelArg( kernel, arg++, sizeof(n        ), &n         );
            err |= clSetKernelArg( kernel, arg++, sizeof(SA       ), &SA        );
            err |= clSetKernelArg( kernel, arg++, sizeof(SA_offset), &SA_offset );
            err |= clSetKernelArg( kernel, arg++, sizeof(ldsa     ), &ldsa      );
            err |= clSetKernelArg( kernel, arg++, sizeof(A        ), &A         );
            err |= clSetKernelArg( kernel, arg++, sizeof(A_offset ), &A_offset  );
            err |= clSetKernelArg( kernel, arg++, sizeof(lda      ), &lda       );
            check_error( err );

            err = clEnqueueNDRangeKernel( queue, kernel, ndim, NULL, grid, threads, 0, NULL, NULL );
            check_error( err );
        }
    }
    else if (uplo == MagmaUpper) {
        kernel = g_runtime.get_kernel( "clat2z_upper" );
        if ( kernel != NULL ) {
            err = 0;
            arg = 0;
            err |= clSetKernelArg( kernel, arg++, sizeof(n        ), &n         );
            err |= clSetKernelArg( kernel, arg++, sizeof(SA       ), &SA        );
            err |= clSetKernelArg( kernel, arg++, sizeof(SA_offset), &SA_offset );
            err |= clSetKernelArg( kernel, arg++, sizeof(ldsa     ), &ldsa      );
            err |= clSetKernelArg( kernel, arg++, sizeof(A        ), &A         );
            err |= clSetKernelArg( kernel, arg++, sizeof(A_offset ), &A_offset  );
            err |= clSetKernelArg( kernel, arg++, sizeof(lda      ), &lda       );
            check_error( err );

            err = clEnqueueNDRangeKernel( queue, kernel, ndim, NULL, grid, threads, 0, NULL, NULL );
            check_error( err );
        }
    }
}
