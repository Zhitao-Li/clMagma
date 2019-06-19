/*
    -- clMAGMA (version 0.4) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c

*/
#include "common_magma.h"

/**
    Purpose
    -------
    ZLAUUM computes the product U * U' or L' * L, where the triangular
    factor U or L is stored in the upper or lower triangular part of
    the array dA.

    If UPLO = MagmaUpper then the upper triangle of the result is stored,
    overwriting the factor U in dA.
    If UPLO = MagmaLower then the lower triangle of the result is stored,
    overwriting the factor L in dA.
    This is the blocked form of the algorithm, calling Level 3 BLAS.

    Arguments
    ---------
    @param[in]
    uplo    magma_uplo_t
            Specifies whether the triangular factor stored in the array dA
            is upper or lower triangular:
      -     = MagmaUpper:  Upper triangular
      -     = MagmaLower:  Lower triangular

    @param[in]
    n       INTEGER
            The order of the triangular factor U or L.  N >= 0.

    @param[in,out]
    dA      DOUBLE PRECISION array on the GPU, dimension (LDDA,N)
            On entry, the triangular factor U or L.
            On exit, if UPLO = MagmaUpper, the upper triangle of dA is
            overwritten with the upper triangle of the product U * U';
            if UPLO = MagmaLower, the lower triangle of dA is overwritten with
            the lower triangle of the product L' * L.

    @param[in]
    ldda    INTEGER
            The leading dimension of the array A.  LDDA >= max(1,N).

    @param[out]
    info    INTEGER
      -     = 0: successful exit
      -     < 0: if INFO = -k, the k-th argument had an illegal value

    @ingroup magma_zposv_aux
    ***************************************************************************/
extern "C" magma_int_t
magma_zlauum_gpu(
    magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex_ptr dA, size_t dA_offset, magma_int_t ldda,
    magma_queue_t queue,
    magma_int_t *info)
{
    // returns cl_mem and offset as 2 values
    #define dA(i_, j_) dA, (dA_offset + (i_) + (j_)*ldda)

    /* Local variables */
    const char* uplo_ = lapack_uplo_const( uplo );
    magma_int_t         nb, i, ib;
    double              d_one = MAGMA_D_ONE;
    magmaDoubleComplex  c_one = MAGMA_Z_ONE;
    magmaDoubleComplex  *work;

    int upper  = (uplo == MagmaUpper);

    *info = 0;

    if (! upper && uplo != MagmaLower)
        *info = -1;
    else if (n < 0)
        *info = -2;
    else if (ldda < max(1,n))
        *info = -4;

    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    nb = magma_get_zpotrf_nb(n);

    if (MAGMA_SUCCESS != magma_zmalloc_cpu( &work, nb*nb )) {
        *info = MAGMA_ERR_HOST_ALLOC;
        return *info;
    }

    if (nb <= 1 || nb >= n) {
        magma_zgetmatrix( n, n, dA(0,0), ldda, work, n, queue );
        lapackf77_zlauum( uplo_, &n, work, &n, info );
        magma_zsetmatrix( n, n, work, n, dA(0,0), ldda, queue );
    }
    else {
        if (upper) {
            /* Compute inverse of upper triangular matrix */
            for (i=0; i < n; i += nb) {
                ib = min(nb, n-i);

                /* Compute the product U * U'. */
                magma_ztrmm( MagmaRight, MagmaUpper,
                             MagmaConjTrans, MagmaNonUnit, i, ib,
                             c_one, dA(i,i), ldda, dA(0, i),ldda, queue );

                magma_zgetmatrix( ib, ib,
                                  dA(i, i), ldda,
                                  work, ib, queue );

                lapackf77_zlauum( MagmaUpperStr, &ib, work, &ib, info );

                magma_zsetmatrix( ib, ib,
                                  work, ib,
                                  dA(i, i), ldda, queue );

                if (i+ib < n) {
                    magma_zgemm( MagmaNoTrans, MagmaConjTrans,
                                 i, ib, (n-i-ib), c_one, dA(0,i+ib),
                                 ldda, dA(i, i+ib), ldda, c_one,
                                 dA(0,i), ldda, queue);

                    magma_zherk( MagmaUpper, MagmaNoTrans, ib,(n-i-ib),
                                 d_one, dA(i, i+ib), ldda,
                                 d_one, dA(i, i),    ldda, queue);
                }
            }
        }
        else {
            /* Compute the product L' * L. */
            for (i=0; i < n; i += nb) {
                ib = min(nb, n-i);

                magma_ztrmm( MagmaLeft, MagmaLower,
                             MagmaConjTrans, MagmaNonUnit, ib,
                             i, c_one, dA(i,i), ldda,
                             dA(i, 0),ldda, queue);

                magma_zgetmatrix( ib, ib,
                                  dA(i, i), ldda,
                                  work, ib, queue );

                lapackf77_zlauum( MagmaLowerStr, &ib, work, &ib, info );

                magma_zsetmatrix( ib, ib,
                                  work, ib,
                                  dA(i, i), ldda, queue );

                if (i+ib < n) {
                    magma_zgemm( MagmaConjTrans, MagmaNoTrans,
                                 ib, i, (n-i-ib), c_one, dA( i+ib,i),
                                 ldda, dA(i+ib, 0),ldda, c_one,
                                 dA(i,0), ldda, queue);
                    magma_zherk( MagmaLower, MagmaConjTrans, ib, (n-i-ib),
                                 d_one, dA(i+ib, i), ldda,
                                 d_one, dA(i, i),    ldda, queue);
                }
            }
        }
    }


    magma_free_cpu( work );

    return *info;
}
