/*
     -- clMAGMA (version 0.1) --
        Univ. of Tennessee, Knoxville
        Univ. of California, Berkeley
        Univ. of Colorado, Denver
        @date
  
       @author Stan Tomov
       @author Mark Gates
        @generated from src/zpotrf_gpu.cpp, normal z -> c, Tue Jun 18 16:14:15 2019
*/
#include "common_magma.h"

/**
    Purpose
    -------
    CPOTRF computes the Cholesky factorization of a complex Hermitian
    positive definite matrix dA.

    The factorization has the form
        dA = U**H * U,   if UPLO = MagmaUpper, or
        dA = L  * L**H,  if UPLO = MagmaLower,
    where U is an upper triangular matrix and L is lower triangular.

    This is the block version of the algorithm, calling Level 3 BLAS.

    Arguments
    ---------
    @param[in]
    uplo    magma_uplo_t
      -     = MagmaUpper:  Upper triangle of dA is stored;
      -     = MagmaLower:  Lower triangle of dA is stored.

    @param[in]
    n       INTEGER
            The order of the matrix dA.  N >= 0.

    @param[in,out]
    dA      COMPLEX array on the GPU, dimension (LDDA,N)
            On entry, the Hermitian matrix dA.  If UPLO = MagmaUpper, the leading
            N-by-N upper triangular part of dA contains the upper
            triangular part of the matrix dA, and the strictly lower
            triangular part of dA is not referenced.  If UPLO = MagmaLower, the
            leading N-by-N lower triangular part of dA contains the lower
            triangular part of the matrix dA, and the strictly upper
            triangular part of dA is not referenced.
    \n
            On exit, if INFO = 0, the factor U or L from the Cholesky
            factorization dA = U**H * U or dA = L * L**H.

    @param[in]
    ldda     INTEGER
            The leading dimension of the array dA.  LDDA >= max(1,N).
            To benefit from coalescent memory accesses LDDA must be
            divisible by 16.

    @param[out]
    info    INTEGER
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
      -     > 0:  if INFO = i, the leading minor of order i is not
                  positive definite, and the factorization could not be
                  completed.

    @ingroup magma_cposv_comp
    ********************************************************************/
extern "C" magma_int_t
magma_cpotrf_gpu(
    magma_uplo_t   uplo, magma_int_t    n,
    magmaFloatComplex_ptr dA, size_t dA_offset, magma_int_t ldda,
    magma_queue_t queue,
    magma_int_t*   info )
{
    #ifdef HAVE_clBLAS
    #define dA(i_, j_)  dA, ((i_) + (j_)*ldda + dA_offset)
    #else
    #define dA(i_, j_) (dA + (i_) + (j_)*ldda)
    #endif

    magma_int_t j, jb, nb;
    const char* uplo_ = lapack_uplo_const( uplo );
    magmaFloatComplex c_one     = MAGMA_C_ONE;
    magmaFloatComplex c_neg_one = MAGMA_C_NEG_ONE;
    magmaFloatComplex *work;
    float    d_one =  1.0;
    float  d_neg_one = -1.0;
    int upper = (uplo == MagmaUpper);
    
    *info = 0;
    if (! upper && uplo != MagmaLower) {
        *info = -1;
    } else if ( n < 0 ) {
        *info = -2;
    } else if ( ldda < max(1,n) ) {
        *info = -4;
    }
    if ( *info != 0 ) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }
    
    nb = magma_get_cpotrf_nb( n );
    
    if (MAGMA_SUCCESS != magma_cmalloc_cpu( &work, nb*nb )) {
        *info = MAGMA_ERR_HOST_ALLOC;
        return *info;
    }
    
    magma_event_t event = NULL;
    
    if ((nb <= 1) || (nb >= n)) {
        // Use unblocked code
        magma_cgetmatrix_async( n, n, dA(0,0), ldda, work, n, queue, NULL );
        magma_queue_sync( queue );
        lapackf77_cpotrf( uplo_, &n, work, &n, info );
        magma_csetmatrix_async( n, n, work, n, dA(0,0), ldda, queue, NULL );
    }
    else {
        // Use blocked code.
        if (upper) {
            // --------------------
            // Compute the Cholesky factorization A = U'*U.
            for( j = 0; j < n; j += nb ) {
                // apply all previous updates to diagonal block,
                // then transfer it to CPU
                jb = min( nb, n-j );
                magma_cherk( MagmaUpper, MagmaConjTrans, jb, j,
                             d_neg_one, dA(0,j), ldda,
                             d_one,     dA(j,j), ldda, queue );
                
                magma_cgetmatrix_async( jb, jb,
                                        dA(j,j), ldda,
                                        work, jb, queue, &event );

                // apply all previous updates to block row right of diagonal block
                if ( j+jb < n ) {
                    magma_cgemm( MagmaConjTrans, MagmaNoTrans,
                                 jb, n-j-jb, j,
                                 c_neg_one, dA(0, j   ), ldda,
                                            dA(0, j+jb), ldda,
                                 c_one,     dA(j, j+jb), ldda, queue );
                }
                
                // simultaneous with above cgemm, transfer diagonal block,
                // factor it on CPU, and test for positive definiteness
                magma_event_sync( event );
                lapackf77_cpotrf( MagmaUpperStr, &jb, work, &jb, info );
                if ( *info != 0 ) {
                    *info = *info + j;
                    break;
                }
                magma_csetmatrix_async( jb, jb,
                                        work, jb,
                                        dA(j,j), ldda, queue, &event );
                
                // apply diagonal block to block row right of diagonal block
                if ( j+jb < n ) {
                    //magma_event_sync( event );
                    magma_ctrsm( MagmaLeft, MagmaUpper, MagmaConjTrans, MagmaNonUnit,
                                 jb, n-j-jb,
                                 c_one, dA(j, j),    ldda,
                                        dA(j, j+jb), ldda, queue );
                }
            }
        }
        else {
            // --------------------
            // Compute the Cholesky factorization A = L*L'.
            for( j = 0; j < n; j += nb ) {
                // apply all previous updates to diagonal block,
                // then transfer it to CPU
                jb = min( nb, n-j );
                magma_cherk( MagmaLower, MagmaNoTrans, jb, j,
                             d_neg_one, dA(j, 0), ldda,
                             d_one,     dA(j, j), ldda, queue );
                
                magma_cgetmatrix_async( jb, jb,
                                        dA(j,j), ldda,
                                        work, jb, queue, &event );

                // apply all previous updates to block column below diagonal block
                if ( j+jb < n ) {
                    magma_cgemm( MagmaNoTrans, MagmaConjTrans,
                                 n-j-jb, jb, j,
                                 c_neg_one, dA(j+jb, 0), ldda,
                                            dA(j,    0), ldda,
                                 c_one,     dA(j+jb, j), ldda, queue );
                }
                
                // simultaneous with above cgemm, transfer diagonal block,
                // factor it on CPU, and test for positive definiteness
                magma_event_sync( event );
                lapackf77_cpotrf( MagmaLowerStr, &jb, work, &jb, info );
                if ( *info != 0 ) {
                    *info = *info + j;
                    break;
                }
                magma_csetmatrix_async( jb, jb,
                                        work, jb,
                                        dA(j,j), ldda, queue, &event );
                
                // apply diagonal block to block column below diagonal
                if ( j+jb < n ) {
                    //magma_event_sync( event );
                    magma_ctrsm( MagmaRight, MagmaLower, MagmaConjTrans, MagmaNonUnit,
                                 n-j-jb, jb,
                                 c_one, dA(j, j   ), ldda,
                                        dA(j+jb, j), ldda, queue );
                }
            }
        }
    }
    
    magma_queue_sync( queue );
    magma_free_cpu( work );
    
    return *info;
}
