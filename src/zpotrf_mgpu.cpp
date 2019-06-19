/*
    -- MAGMA (version 1.3.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c

*/
#include "common_magma.h"

/* use two queues; one for comm, and one for comp */
extern "C" magma_int_t
magma_zpotrf2_mgpu(
    magma_int_t num_gpus, magma_uplo_t uplo, magma_int_t m, magma_int_t n, 
    magma_int_t off_i, magma_int_t off_j, magma_int_t nb,
    magmaDoubleComplex_ptr *d_lA, size_t dA_offset, magma_int_t ldda, 
    magmaDoubleComplex_ptr *d_lP, magma_int_t lddp,
    magmaDoubleComplex  *a,    magma_int_t lda, magma_int_t h,
    magma_queue_t *queues,
    magma_int_t *info );


extern "C" magma_int_t
magma_zpotrf_mgpu(
    magma_int_t num_gpus, magma_uplo_t uplo, magma_int_t n, 
    magmaDoubleComplex_ptr *d_lA, size_t dA_offset, 
    magma_int_t ldda,
    magma_queue_t *queues,
    magma_int_t *info)
{
/*  -- clMAGMA (version 1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

    Purpose   
    =======   
    ZPOTRF computes the Cholesky factorization of a complex Hermitian   
    positive definite matrix dA.   

    The factorization has the form   
       dA = U**H * U,  if UPLO = 'U', or   
       dA = L  * L**H,  if UPLO = 'L',   
    where U is an upper triangular matrix and L is lower triangular.   

    This is the block version of the algorithm, calling Level 3 BLAS.   

    Arguments   
    =========   
    UPLO    (input) CHARACTER*1   
            = 'U':  Upper triangle of dA is stored;   
            = 'L':  Lower triangle of dA is stored.   

    N       (input) INTEGER   
            The order of the matrix dA.  N >= 0.   

    dA      (input/output) COMPLEX_16 array on the GPU, dimension (LDDA,N)   
            On entry, the Hermitian matrix dA.  If UPLO = 'U', the leading   
            N-by-N upper triangular part of dA contains the upper   
            triangular part of the matrix dA, and the strictly lower   
            triangular part of dA is not referenced.  If UPLO = 'L', the   
            leading N-by-N lower triangular part of dA contains the lower   
            triangular part of the matrix dA, and the strictly upper   
            triangular part of dA is not referenced.   

            On exit, if INFO = 0, the factor U or L from the Cholesky   
            factorization dA = U**H * U or dA = L * L**H.   

    LDDA     (input) INTEGER   
            The leading dimension of the array dA.  LDDA >= max(1,N).
            To benefit from coalescent memory accesses LDDA must be
            divisible by 16.

    INFO    (output) INTEGER   
            = 0:  successful exit   
            < 0:  if INFO = -i, the i-th argument had an illegal value   
            > 0:  if INFO = i, the leading minor of order i is not   
                  positive definite, and the factorization could not be   
                  completed.   
    =====================================================================   */

    magma_int_t     j, nb, d, lddp, h;
    magmaDoubleComplex *work;
    magmaDoubleComplex_ptr dwork[MagmaMaxGPUs];

    *info = 0;
    nb = magma_get_zpotrf_nb(n);
    if ( uplo != MagmaUpper && uplo != MagmaLower ) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (uplo != MagmaUpper) {
        lddp = nb*(n/(nb*num_gpus));
        if( n%(nb*num_gpus) != 0 ) lddp+=min(nb,n-num_gpus*lddp);
        if( ldda < lddp ) *info = -4;
    } else if( ldda < n ) {
        *info = -4;
    }
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    magma_int_t err;
    
    if (num_gpus == 1 && ((nb <= 1) || (nb >= n)) ) {
      /*  Use unblocked code. */
        err = magma_zmalloc_cpu( &work, n*nb );
        if (err != MAGMA_SUCCESS) {
        *info = MAGMA_ERR_HOST_ALLOC;
        return *info;
      }
      magma_zgetmatrix( n, n, d_lA[0], 0, ldda, work, n, queues[0] );
      lapackf77_zpotrf(lapack_uplo_const(uplo), &n, work, &n, info);
      magma_zsetmatrix( n, n, work, n, d_lA[0], 0, ldda, queues[0] );
      magma_free_cpu( work );
    } else {
      lddp = 32*((n+31)/32);
      for( d=0; d<num_gpus; d++ ) {
        if (MAGMA_SUCCESS != magma_zmalloc( &dwork[d], num_gpus*nb*lddp )) {
          for( j=0; j<d; j++ ) {
            magma_free( dwork[j] );
          }
          *info = MAGMA_ERR_DEVICE_ALLOC;
          return *info;
        }
      }
      h = 1; //num_gpus; //magma_ceildiv( n, nb );
      if (MAGMA_SUCCESS != magma_zmalloc_cpu( &work, n*nb*h )) {
        *info = MAGMA_ERR_HOST_ALLOC;
        return *info;
      }
      if (uplo == MagmaUpper) {
          /* with two queues for each device */
          magma_zpotrf2_mgpu(num_gpus, uplo, n, n, 0, 0, nb, d_lA, 0, ldda, dwork, lddp, work, n, h, queues, info);
          /* with three streams */
          //magma_zpotrf3_mgpu(num_gpus, uplo, n, n, 0, 0, nb, d_lA, ldda, dwork, lddp, work, n,  
          //                   h, stream, event, info);
      } else {
          /* with two queues for each device */
          magma_zpotrf2_mgpu(num_gpus, uplo, n, n, 0, 0, nb, d_lA, 0, ldda, dwork, lddp, work, nb*h, h, queues, info);
          /* with three streams */
          //magma_zpotrf3_mgpu(num_gpus, uplo, n, n, 0, 0, nb, d_lA, ldda, dwork, lddp, work, nb*h, 
          //                   h, stream, event, info);
      }

      /* clean up */
      for( d=0; d<num_gpus; d++ ) {
        //magma_queue_sync( queues[d] );
        magma_free( dwork[d] );
      }
      magma_free_cpu( work );
    } /* end of not lapack */


    return *info;
} /* magma_zpotrf_mgpu */
