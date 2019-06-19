/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c

       auto-converted from zlaswp.cu
       
       @author Stan Tomov
       @author Mathieu Faverge
       @author Ichitaro Yamazaki
       @author Mark Gates
*/
#include "clmagma_runtime.h"
#include "common_magma.h"
#include "zlaswp.h"


/**
    Purpose:
    =============
    ZLASWP performs a series of row interchanges on the matrix A.
    One row interchange is initiated for each of rows K1 through K2 of A.
    
    ** Unlike LAPACK, here A is stored row-wise (hence dAT). **
    Otherwise, this is identical to LAPACK's interface.
    
    Arguments:
    ==========
    @param[in]
    n       INTEGER
            The number of columns of the matrix A.
    
    @param[in,out]
    dAT     COMPLEX*16 array on GPU, stored row-wise, dimension (LDDA,N)
            On entry, the matrix of column dimension N to which the row
            interchanges will be applied.
            On exit, the permuted matrix.
    
    @param[in]
    ldda    INTEGER
            The leading dimension of the array A. ldda >= n.
    
    @param[in]
    k1      INTEGER
            The first element of IPIV for which a row interchange will
            be done. (Fortran one-based index: 1 <= k1 .)
    
    @param[in]
    k2      INTEGER
            The last element of IPIV for which a row interchange will
            be done. (Fortran one-based index: 1 <= k2 .)
    
    @param[in]
    ipiv    INTEGER array, on CPU, dimension (K2*abs(INCI))
            The vector of pivot indices.  Only the elements in positions
            K1 through K2 of IPIV are accessed.
            IPIV(K) = L implies rows K and L are to be interchanged.
    
    @param[in]
    inci    INTEGER
            The increment between successive values of IPIV.
            Currently, INCI > 0.
            TODO: If INCI is negative, the pivots are applied in reverse order.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_zaux2
    ********************************************************************/
// It is used in zgessm, zgetrf_incpiv.
extern "C" void
magmablas_zlaswp(
    magma_int_t n,
    magmaDoubleComplex_ptr dAT, size_t dAT_offset, magma_int_t ldda,
    magma_int_t k1, magma_int_t k2,
    const magma_int_t *ipiv, magma_int_t inci,
    magma_queue_t queue )
{
    cl_kernel kernel;
    cl_int err;
    int arg;

    magma_int_t info = 0;
    if ( n < 0 )
        info = -1;
    else if ( n > ldda )
        info = -3;
    else if ( k1 < 1 )
        info = -4;
    else if ( k2 < 1 )
        info = -5;
    else if ( inci <= 0 )
        info = -7;

    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;  //info;
    }
    
    const int ndim = 1;
    size_t threads[ndim];
    threads[0] = NTHREADS;
    size_t grid[ndim];
    grid[0] = magma_ceildiv( n, NTHREADS );
    grid[0] *= threads[0];
    zlaswp_params_t params;
    
    kernel = g_runtime.get_kernel( "zlaswp_kernel" );
    
    for( int k = k1-1; k < k2; k += MAX_PIVOTS ) {
        int npivots = min( MAX_PIVOTS, k2-k );
        params.npivots = npivots;
        for( int j = 0; j < npivots; ++j ) {
            params.ipiv[j] = ipiv[(k+j)*inci] - k - 1;
        }
        if ( kernel != NULL ) {
            err = 0;
            arg = 0;
            size_t k_offset = dAT_offset + k*ldda;
            err |= clSetKernelArg( kernel, arg++, sizeof(n              ), &n               );
            err |= clSetKernelArg( kernel, arg++, sizeof(dAT            ), &dAT             );
            err |= clSetKernelArg( kernel, arg++, sizeof(k_offset), &k_offset );
            err |= clSetKernelArg( kernel, arg++, sizeof(ldda           ), &ldda            );
            err |= clSetKernelArg( kernel, arg++, sizeof(params         ), &params          );
            check_error( err );

            err = clEnqueueNDRangeKernel( queue, kernel, ndim, NULL, grid, threads, 0, NULL, NULL );
            check_error( err );
        }
    }
}


/**
    Purpose:
    =============
    ZLASWPX performs a series of row interchanges on the matrix A.
    One row interchange is initiated for each of rows K1 through K2 of A.
    
    ** Unlike LAPACK, here A is stored either row-wise or column-wise,
       depending on ldx and ldy. **
    Otherwise, this is identical to LAPACK's interface.
    
    Arguments:
    ==========
    @param[in]
    n        INTEGER
             The number of columns of the matrix A.
    
    @param[in,out]
    dA       COMPLEX*16 array on GPU, dimension (*,*)
             On entry, the matrix of column dimension N to which the row
             interchanges will be applied.
             On exit, the permuted matrix.
    
    @param[in]
    ldx      INTEGER
             Stride between elements in same column.
    
    @param[in]
    ldy      INTEGER
             Stride between elements in same row.
             For A stored row-wise,    set ldx=ldda and ldy=1.
             For A stored column-wise, set ldx=1    and ldy=ldda.
    
    @param[in]
    k1       INTEGER
             The first element of IPIV for which a row interchange will
             be done. (One based index.)
    
    @param[in]
    k2       INTEGER
             The last element of IPIV for which a row interchange will
             be done. (One based index.)
    
    @param[in]
    ipiv     INTEGER array, on CPU, dimension (K2*abs(INCI))
             The vector of pivot indices.  Only the elements in positions
             K1 through K2 of IPIV are accessed.
             IPIV(K) = L implies rows K and L are to be interchanged.
    
    @param[in]
    inci     INTEGER
             The increment between successive values of IPIV.
             Currently, IPIV > 0.
             TODO: If IPIV is negative, the pivots are applied in reverse order.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_zaux2
    ********************************************************************/
extern "C" void
magmablas_zlaswpx(
    magma_int_t n,
    magmaDoubleComplex_ptr dA, size_t dA_offset, magma_int_t ldx, magma_int_t ldy,
    magma_int_t k1, magma_int_t k2,
    const magma_int_t *ipiv, magma_int_t inci,
    magma_queue_t queue )
{
    cl_kernel kernel;
    cl_int err;
    int arg;

    magma_int_t info = 0;
    if ( n < 0 )
        info = -1;
    else if ( k1 < 0 )
        info = -4;
    else if ( k2 < 0 || k2 < k1 )
        info = -5;
    else if ( inci <= 0 )
        info = -7;

    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;  //info;
    }
    
    const int ndim = 1;
    size_t threads[ndim];
    threads[0] = NTHREADS;
    size_t grid[ndim];
    grid[0] = magma_ceildiv( n, NTHREADS );
    grid[0] *= threads[0];
    zlaswp_params_t params;
    
    kernel = g_runtime.get_kernel( "zlaswpx_kernel" );
    
    for( int k = k1-1; k < k2; k += MAX_PIVOTS ) {
        int npivots = min( MAX_PIVOTS, k2-k );
        params.npivots = npivots;
        for( int j = 0; j < npivots; ++j ) {
            params.ipiv[j] = ipiv[(k+j)*inci] - k - 1;
        }
        if ( kernel != NULL ) {
            err = 0;
            arg = 0;
            size_t k_offset = dA_offset + k*ldx;
            err |= clSetKernelArg( kernel, arg++, sizeof(n             ), &n              );
            err |= clSetKernelArg( kernel, arg++, sizeof(dA            ), &dA             );
            err |= clSetKernelArg( kernel, arg++, sizeof(k_offset), &k_offset );
            err |= clSetKernelArg( kernel, arg++, sizeof(ldx           ), &ldx            );
            err |= clSetKernelArg( kernel, arg++, sizeof(ldy           ), &ldy            );
            err |= clSetKernelArg( kernel, arg++, sizeof(params        ), &params         );
            check_error( err );

            err = clEnqueueNDRangeKernel( queue, kernel, ndim, NULL, grid, threads, 0, NULL, NULL );
            check_error( err );
        }
    }
}


/**
    Purpose:
    =============
    ZLASWP2 performs a series of row interchanges on the matrix A.
    One row interchange is initiated for each of rows K1 through K2 of A.
    
    ** Unlike LAPACK, here A is stored row-wise (hence dAT). **
    Otherwise, this is identical to LAPACK's interface.
    
    Here, d_ipiv is passed in GPU memory.
    
    Arguments:
    ==========
    @param[in]
    n        INTEGER
             The number of columns of the matrix A.
    
    @param[in,out]
    dAT      COMPLEX*16 array on GPU, stored row-wise, dimension (LDDA,*)
             On entry, the matrix of column dimension N to which the row
             interchanges will be applied.
             On exit, the permuted matrix.
    
    @param[in]
    ldda     INTEGER
             The leading dimension of the array A.
             (I.e., stride between elements in a column.)
    
    @param[in]
    k1       INTEGER
             The first element of IPIV for which a row interchange will
             be done. (One based index.)
    
    @param[in]
    k2       INTEGER
             The last element of IPIV for which a row interchange will
             be done. (One based index.)
    
    @param[in]
    d_ipiv   INTEGER array, on GPU, dimension (K2*abs(INCI))
             The vector of pivot indices.  Only the elements in positions
             K1 through K2 of IPIV are accessed.
             IPIV(K) = L implies rows K and L are to be interchanged.
    
    @param[in]
    inci     INTEGER
             The increment between successive values of IPIV.
             Currently, IPIV > 0.
             TODO: If IPIV is negative, the pivots are applied in reverse order.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_zaux2
    ********************************************************************/
extern "C" void
magmablas_zlaswp2(
    magma_int_t n,
    magmaDoubleComplex_ptr dAT, size_t dAT_offset, magma_int_t ldda,
    magma_int_t k1, magma_int_t k2,
    magmaInt_const_ptr d_ipiv, size_t d_ipiv_offset, magma_int_t inci,
    magma_queue_t queue )
{
    cl_kernel kernel;
    cl_int err;
    int arg;

    magma_int_t info = 0;
    if ( n < 0 )
        info = -1;
    else if ( k1 < 0 )
        info = -4;
    else if ( k2 < 0 || k2 < k1 )
        info = -5;
    else if ( inci <= 0 )
        info = -7;

    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;  //info;
    }
    
    magma_int_t nb = k2-(k1-1);
    
    const int ndim = 1;
    size_t threads[ndim];
    threads[0] = NTHREADS;
    size_t grid[ndim];
    grid[0] = magma_ceildiv( n, NTHREADS );
    grid[0] *= threads[0];
    kernel = g_runtime.get_kernel( "zlaswp2_kernel" );
    if ( kernel != NULL ) {
        err = 0;
        arg = 0;
        err |= clSetKernelArg( kernel, arg++, sizeof(n                 ), &n                  );
        err |= clSetKernelArg( kernel, arg++, sizeof(dAT               ), &dAT                );
        err |= clSetKernelArg( kernel, arg++, sizeof(dAT_offset   ), &dAT_offset    );
        err |= clSetKernelArg( kernel, arg++, sizeof(ldda              ), &ldda               );
        err |= clSetKernelArg( kernel, arg++, sizeof(nb                ), &nb                 );
        err |= clSetKernelArg( kernel, arg++, sizeof(d_ipiv            ), &d_ipiv             );
        err |= clSetKernelArg( kernel, arg++, sizeof(d_ipiv_offset     ), &d_ipiv_offset      );
        err |= clSetKernelArg( kernel, arg++, sizeof(inci              ), &inci               );
        check_error( err );

        err = clEnqueueNDRangeKernel( queue, kernel, ndim, NULL, grid, threads, 0, NULL, NULL );
        check_error( err );
    }
}
