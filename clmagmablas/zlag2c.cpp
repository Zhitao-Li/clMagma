/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions mixed zc -> ds

       auto-converted from zlag2c.cu
       @author Mark Gates
*/
#include "clmagma_runtime.h"
#include "common_magma.h"
#include "zlag2c.h"


/**
    Purpose
    -------
    ZLAG2C converts a double-complex matrix, A,
                 to a single-complex matrix, SA.
    
    RMAX is the overflow for the single-complex arithmetic.
    ZLAG2C checks that all the entries of A are between -RMAX and
    RMAX. If not, the conversion is aborted and a flag is raised.
    
    Arguments
    ---------
    @param[in]
    m       INTEGER
            The number of lines of the matrix A.  m >= 0.
    
    @param[in]
    n       INTEGER
            The number of columns of the matrix A.  n >= 0.
    
    @param[in]
    A       COMPLEX_16 array, dimension (LDA,n)
            On entry, the m-by-n coefficient matrix A.
    
    @param[in]
    lda     INTEGER
            The leading dimension of the array A.  LDA >= max(1,m).
    
    @param[out]
    SA      COMPLEX array, dimension (LDSA,n)
            On exit, if INFO=0, the m-by-n coefficient matrix SA;
            if INFO > 0, the content of SA is unspecified.
    
    @param[in]
    ldsa    INTEGER
            The leading dimension of the array SA.  LDSA >= max(1,m).
    
    @param[out]
    info    INTEGER
      -     = 0:  successful exit.
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
      -     = 1:  an entry of the matrix A is greater than the COMPLEX
                  overflow threshold, in this case, the content
                  of SA on exit is unspecified.
    
    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_zaux2
    ********************************************************************/
extern "C" void
magmablas_zlag2c(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_const_ptr A, size_t A_offset, magma_int_t lda,
    magmaFloatComplex_ptr SA, size_t SA_offset,       magma_int_t ldsa,
    magma_queue_t queue,
    magma_int_t *info )
{
    cl_kernel kernel;
    cl_int err;
    int arg;

    *info = 0;
    if ( m < 0 )
        *info = -1;
    else if ( n < 0 )
        *info = -2;
    else if ( lda < max(1,m) )
        *info = -4;
    else if ( ldsa < max(1,m) )
        *info = -6;
    
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return; //*info;
    }

    /* quick return */
    if ( m == 0 || n == 0 ) {
        return;
    }
    
    double rmax = (double)lapackf77_slamch("O");

    const int ndim = 2;
    size_t threads[ndim];
    threads[0] = BLK_X;
    threads[1] = 1;
    size_t grid[ndim];
    grid[0] = magma_ceildiv( m, BLK_X );
    grid[1] = magma_ceildiv( n, BLK_Y );
    grid[0] *= threads[0];
    grid[1] *= threads[1];
    
    // TODO cudaMemcpyToSymbol( flag, info, sizeof(flag) );    // flag = 0
    
    kernel = g_runtime.get_kernel( "zlag2c_kernel" );
    if ( kernel != NULL ) {
        err = 0;
        arg = 0;
        err |= clSetKernelArg( kernel, arg++, sizeof(m        ), &m         );
        err |= clSetKernelArg( kernel, arg++, sizeof(n        ), &n         );
        err |= clSetKernelArg( kernel, arg++, sizeof(A        ), &A         );
        err |= clSetKernelArg( kernel, arg++, sizeof(A_offset ), &A_offset  );
        err |= clSetKernelArg( kernel, arg++, sizeof(lda      ), &lda       );
        err |= clSetKernelArg( kernel, arg++, sizeof(SA       ), &SA        );
        err |= clSetKernelArg( kernel, arg++, sizeof(SA_offset), &SA_offset );
        err |= clSetKernelArg( kernel, arg++, sizeof(ldsa     ), &ldsa      );
        err |= clSetKernelArg( kernel, arg++, sizeof(rmax     ), &rmax      );
        check_error( err );

        err = clEnqueueNDRangeKernel( queue, kernel, ndim, NULL, grid, threads, 0, NULL, NULL );
        check_error( err );
    }
    
    // TODO cudaMemcpyFromSymbol( info, flag, sizeof(flag) );  // info = flag
}
