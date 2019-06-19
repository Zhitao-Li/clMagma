/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @generated from clmagmablas/zlascl_2x2.cpp, normal z -> s, Tue Jun 18 16:14:18 2019

       auto-converted from slascl_2x2.cu

       @author Ichitaro Yamazaki
*/
#include "clmagma_runtime.h"
#include "common_magma.h"
#include "slascl_2x2.h"


/**
    Purpose
    -------
    SLASCL_2x2 scales the M by M real matrix A by the 2-by-2 pivot.
    TYPE specifies that A may be upper or lower triangular.

    Arguments
    ---------
    @param[in]
    type    magma_type_t
            TYPE indices the storage type of the input matrix A.
            = MagmaLower:  lower triangular matrix.
            = MagmaUpper:  upper triangular matrix.
            Other formats that LAPACK supports, MAGMA does not currently support.

    @param[in]
    m       INTEGER
            The number of rows of the matrix A.  M >= 0.

    @param[in]
    dW      REAL vector, dimension (2*lddw)
            The matrix containing the 2-by-2 pivot.

    @param[in]
    lddw    INTEGER
            The leading dimension of the array W.  LDDA >= max(1,M).

    @param[in,out]
    dA      REAL array, dimension (LDDA,N)
            The matrix to be scaled by dW.  See TYPE for the
            storage type.

    @param[in]
    ldda    INTEGER
            The leading dimension of the array A.  LDDA >= max(1,M).

    @param[out]
    info    INTEGER
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value.

    @ingroup magma_saux2
    ********************************************************************/
extern "C" void
magmablas_slascl_2x2(
    magma_type_t type, magma_int_t m,
    magmaFloat_const_ptr dW, size_t dW_offset, magma_int_t lddw,
    magmaFloat_ptr       dA, size_t dA_offset, magma_int_t ldda,
    magma_queue_t queue,
    magma_int_t *info )
{
    cl_kernel kernel;
    cl_int err;
    int arg;

    *info = 0;
    if ( type != MagmaLower && type != MagmaUpper )
        *info = -1;
    else if ( m < 0 )
        *info = -2;
    else if ( ldda < max(1,m) )
        *info = -4;
    
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return;  //info;
    }
    
    const int ndim = 1;
    size_t threads[ndim];
    threads[0] = NB;
    size_t grid[ndim];
    grid[0] = magma_ceildiv( m, NB );
    grid[0] *= threads[0];
    
    if (type == MagmaLower) {
        kernel = g_runtime.get_kernel( "slascl_2x2_lower" );
        if ( kernel != NULL ) {
            err = 0;
            arg = 0;
            err |= clSetKernelArg( kernel, arg++, sizeof(m        ), &m         );
            err |= clSetKernelArg( kernel, arg++, sizeof(dW       ), &dW        );
            err |= clSetKernelArg( kernel, arg++, sizeof(dW_offset), &dW_offset );
            err |= clSetKernelArg( kernel, arg++, sizeof(lddw     ), &lddw      );
            err |= clSetKernelArg( kernel, arg++, sizeof(dA       ), &dA        );
            err |= clSetKernelArg( kernel, arg++, sizeof(dA_offset), &dA_offset );
            err |= clSetKernelArg( kernel, arg++, sizeof(ldda     ), &ldda      );
            check_error( err );

            err = clEnqueueNDRangeKernel( queue, kernel, ndim, NULL, grid, threads, 0, NULL, NULL );
            check_error( err );
        }
    }
    else {
        kernel = g_runtime.get_kernel( "slascl_2x2_upper" );
        if ( kernel != NULL ) {
            err = 0;
            arg = 0;
            err |= clSetKernelArg( kernel, arg++, sizeof(m        ), &m         );
            err |= clSetKernelArg( kernel, arg++, sizeof(dW       ), &dW        );
            err |= clSetKernelArg( kernel, arg++, sizeof(dW_offset), &dW_offset );
            err |= clSetKernelArg( kernel, arg++, sizeof(lddw     ), &lddw      );
            err |= clSetKernelArg( kernel, arg++, sizeof(dA       ), &dA        );
            err |= clSetKernelArg( kernel, arg++, sizeof(dA_offset), &dA_offset );
            err |= clSetKernelArg( kernel, arg++, sizeof(ldda     ), &ldda      );
            check_error( err );

            err = clEnqueueNDRangeKernel( queue, kernel, ndim, NULL, grid, threads, 0, NULL, NULL );
            check_error( err );
        }
    }
}
