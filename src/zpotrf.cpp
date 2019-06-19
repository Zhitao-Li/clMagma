/*
    -- clMAGMA (version 0.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Stan Tomov
       @precisions normal z -> s d c
*/
#include "common_magma.h"


#define A(i, j)  (a   +(j)*lda  + (i))
#define dA(i, j) dwork, ((j)*ldda + (i))

extern "C" magma_int_t
magma_zpotrf(
    magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex *a, magma_int_t lda,
    magma_queue_t queues[2],
    magma_int_t *info )
{
/*  -- clMAGMA (version 0.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

    Purpose
    =======
    ZPOTRF computes the Cholesky factorization of a complex Hermitian
    positive definite matrix A. This version does not require work
    space on the GPU passed as input. GPU memory is allocated in the
    routine.

    The factorization has the form
        A = U**H * U,  if UPLO = 'U', or
        A = L  * L**H, if UPLO = 'L',
    where U is an upper triangular matrix and L is lower triangular.

    This is the block version of the algorithm, calling Level 3 BLAS.
    If the current stream is NULL, this version replaces it with user defined
    stream to overlap computation with communication.

    Arguments
    =========
    UPLO    (input) CHARACTER*1
            = 'U':  Upper triangle of A is stored;
            = 'L':  Lower triangle of A is stored.

    N       (input) INTEGER
            The order of the matrix A.  N >= 0.

    A       (input/output) COMPLEX_16 array, dimension (LDA,N)
            On entry, the Hermitian matrix A.  If UPLO = 'U', the leading
            N-by-N upper triangular part of A contains the upper
            triangular part of the matrix A, and the strictly lower
            triangular part of A is not referenced.  If UPLO = 'L', the
            leading N-by-N lower triangular part of A contains the lower
            triangular part of the matrix A, and the strictly upper
            triangular part of A is not referenced.

            On exit, if INFO = 0, the factor U or L from the Cholesky
            factorization A = U**H * U or A = L * L**H.

            Higher performance is achieved if A is in pinned memory, e.g.
            allocated using magma_malloc_pinned.

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,N).

    INFO    (output) INTEGER
            = 0:  successful exit
            < 0:  if INFO = -i, the i-th argument had an illegal value
                  or another error occured, such as memory allocation failed.
            > 0:  if INFO = i, the leading minor of order i is not
                  positive definite, and the factorization could not be
                  completed.

    =====================================================================    */

    magma_int_t ldda, nb, j, jb;
    magmaDoubleComplex c_one     = MAGMA_Z_ONE;
    magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    magmaDoubleComplex_ptr  dwork;
    double             d_one     =  1.0;
    double             d_neg_one = -1.0;

    *info = 0;
    if( (uplo != MagmaUpper) && (uplo != MagmaLower) ) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (lda < max(1,n)) {
        *info = -4;
    }
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    /* Quick return */
    if ( n == 0 )
        return *info;

    magma_int_t num_gpus = magma_num_gpus();
    if( num_gpus > 1 ) {
        /* call multiple-GPU interface  */
      printf("multiple-GPU verison not implemented\n"); 
        return MAGMA_ERR_NOT_IMPLEMENTED;
        //return magma_zpotrf_m(num_gpus, uplo, n, a, lda, info);
    }

    ldda = magma_roundup( n, 32 );
    
    if (MAGMA_SUCCESS != magma_zmalloc( &dwork, (n)*ldda )) {
        /* alloc failed so call the non-GPU-resident version */
        printf("non-GPU-resident version not implemented\n"); 
        return MAGMA_ERR_NOT_IMPLEMENTED;
        //return magma_zpotrf_m(num_gpus, uplo, n, a, lda, info);
    }

    nb = magma_get_zpotrf_nb(n);

    if (nb <= 1 || nb >= n) {
        lapackf77_zpotrf(lapack_uplo_const(uplo), &n, a, &lda, info);
    } else {
        /* Use hybrid blocked code. */
        if (uplo == MagmaUpper) {
            /* Compute the Cholesky factorization A = U'*U. */
            for (j=0; j<n; j += nb) {
                /* Update and factorize the current diagonal block and test
                   for non-positive-definiteness. Computing MIN */
                jb = min(nb, (n-j));
                magma_zsetmatrix_async( jb, (n-j), A(j, j), lda, dA(j, j), ldda, queues[1], NULL);
                
                magma_zherk(MagmaUpper, MagmaConjTrans, jb, j,
                            d_neg_one, dA(0, j), ldda,
                            d_one,     dA(j, j), ldda, queues[1]);
                magma_queue_sync( queues[1] );

                magma_zgetmatrix_async( jb, jb,
                                        dA(j, j), ldda,
                                        A(j, j), lda, queues[0], NULL );
                
                if ( (j+jb) < n) {
                    magma_zgemm(MagmaConjTrans, MagmaNoTrans,
                                jb, (n-j-jb), j,
                                c_neg_one, dA(0, j   ), ldda,
                                           dA(0, j+jb), ldda,
                                c_one,     dA(j, j+jb), ldda, queues[1]);
                }
                
                magma_queue_sync( queues[0] );
                magma_zgetmatrix_async( j, jb,
                                        dA(0, j), ldda,
                                        A (0, j), lda, queues[0], NULL );

                lapackf77_zpotrf(MagmaUpperStr, &jb, A(j, j), &lda, info);
                if (*info != 0) {
                    *info = *info + j;
                    break;
                }
                magma_zsetmatrix_async( jb, jb,
                                        A(j, j), lda,
                                        dA(j, j), ldda, queues[0], NULL );
                magma_queue_sync( queues[0] );

                if ( (j+jb) < n ) {
                    magma_ztrsm(MagmaLeft, MagmaUpper, MagmaConjTrans, MagmaNonUnit,
                                jb, (n-j-jb),
                                c_one, dA(j, j   ), ldda,
                                dA(j, j+jb), ldda, queues[1] );
                }
            }
        }
        else {
            //=========================================================
            // Compute the Cholesky factorization A = L*L'.
            for (j=0; j<n; j+=nb) {
                //  Update and factorize the current diagonal block and test
                //  for non-positive-definiteness. Computing MIN
                jb = min(nb, (n-j));
                magma_zsetmatrix_async( (n-j), jb, A(j, j), lda, dA(j, j), ldda, queues[1], NULL);

                magma_zherk(MagmaLower, MagmaNoTrans, jb, j,
                            d_neg_one, dA(j, 0), ldda,
                            d_one,     dA(j, j), ldda, queues[1]);
                magma_queue_sync( queues[1] );

                magma_zgetmatrix_async( jb, jb,
                                        dA(j,j), ldda,
                                        A(j,j), lda, queues[0], NULL );

                if ( (j+jb) < n) {
                    magma_zgemm( MagmaNoTrans, MagmaConjTrans,
                                 (n-j-jb), jb, j,
                                 c_neg_one, dA(j+jb, 0), ldda,
                                            dA(j,    0), ldda,
                                 c_one,     dA(j+jb, j), ldda, queues[1]);
                }
                
                magma_queue_sync( queues[0] );
                magma_zgetmatrix_async( jb, j,
                                        dA(j, 0), ldda,
                                        A(j, 0), lda, queues[1], NULL );

                lapackf77_zpotrf(MagmaLowerStr, &jb, A(j, j), &lda, info);
                if (*info != 0){
                    *info = *info + j;
                    break;
                } 
                magma_zsetmatrix_async( jb, jb,
                                        A(j, j), lda,
                                        dA(j, j), ldda, queues[0], NULL );
                magma_queue_sync( queues[0] );

                if ( (j+jb) < n) {
                    magma_ztrsm(MagmaRight, MagmaLower, MagmaConjTrans, MagmaNonUnit,
                                (n-j-jb), jb,
                                c_one, dA(j,    j), ldda,
                                dA(j+jb, j), ldda, queues[1]);
                }
            }
        }
    }
    
    magma_free( dwork );
    
    return *info;
} /* magma_zpotrf */
