/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @generated from clmagmablas/clag2z.cpp, mixed zc -> ds, Tue Jun 18 16:14:18 2019

       auto-converted from slag2d.cu
       @author Mark Gates
*/
#include "clmagma_runtime.h"
#include "common_magma.h"
#include "slag2d.h"


/**
    Purpose
    -------
    SLAG2D_STREAM converts a single-real matrix, SA,
                        to a double-real matrix, A.

    Note that while it is possible to overflow while converting
    from double to single, it is not possible to overflow when
    converting from single to double.

    Arguments
    ---------
    @param[in]
    m       INTEGER
            The number of lines of the matrix A.  M >= 0.

    @param[in]
    n       INTEGER
            The number of columns of the matrix A.  N >= 0.

    @param[in]
    SA      SINGLE PRECISION array, dimension (LDSA,N)
            On entry, the M-by-N coefficient matrix SA.

    @param[in]
    ldsa    INTEGER
            The leading dimension of the array SA.  LDSA >= max(1,M).

    @param[out]
    A       DOUBLE PRECISION array, dimension (LDA,N)
            On exit, the M-by-N coefficient matrix A.

    @param[in]
    lda     INTEGER
            The leading dimension of the array A.  LDA >= max(1,M).

    @param[out]
    info    INTEGER
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
    
    @param[in]
    queue   magma_queue_t
            Queue to execute in.
    
    @ingroup magma_saux2
    ********************************************************************/
extern "C" void
magmablas_slag2d(
    magma_int_t m, magma_int_t n,
    magmaFloat_const_ptr SA, size_t SA_offset, magma_int_t ldsa,
    magmaDouble_ptr       A, size_t A_offset, magma_int_t lda,
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
    else if ( ldsa < max(1,m) )
        *info = -4;
    else if ( lda < max(1,m) )
        *info = -6;

    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return; //*info;
    }

    /* quick return */
    if ( m == 0 || n == 0 ) {
        return;
    }

    const int ndim = 2;
    size_t threads[ndim];
    threads[0] = BLK_X;
    threads[1] = 1;
    size_t grid[ndim];
    grid[0] = magma_ceildiv( m, BLK_X );
    grid[1] = magma_ceildiv( n, BLK_Y );
    grid[0] *= threads[0];
    grid[1] *= threads[1];
    kernel = g_runtime.get_kernel( "slag2d_kernel" );
    if ( kernel != NULL ) {
        err = 0;
        arg = 0;
        err |= clSetKernelArg( kernel, arg++, sizeof(m        ), &m         );
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
