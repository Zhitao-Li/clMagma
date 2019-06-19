/*
    -- MAGMA (version 1.3.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c

*/
#include "common_magma.h"

#define USE_PINNED_CLMEMORY
#ifdef  USE_PINNED_CLMEMORY
extern cl_context gContext;
#endif

extern "C" magma_int_t
magma_zpotrf_msub(
    magma_int_t num_subs, magma_int_t num_gpus, magma_uplo_t uplo, magma_int_t n, 
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

    int tot_subs = num_subs * num_gpus;
    magma_int_t err;
    magma_int_t j, nb, d, lddp, h;
    magmaDoubleComplex *work;
    magmaDoubleComplex_ptr dwork[MagmaMaxGPUs];

    *info = 0;
    nb = magma_get_zpotrf_nb(n);
    if ( uplo != MagmaUpper && uplo != MagmaLower ) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (uplo != MagmaUpper) {
        lddp = nb*(n/(nb*tot_subs));
        if( n%(nb*tot_subs) != 0 ) lddp+=min(nb,n-tot_subs*lddp);
        if( ldda < lddp ) *info = -4;
    } else if( ldda < n ) {
        *info = -4;
    }
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

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
        for (d=0; d<num_gpus; d++) {
            if (MAGMA_SUCCESS != magma_zmalloc( &dwork[d], num_gpus*nb*lddp )) {
                for( j=0; j<d; j++ ) magma_free( dwork[j] );
                *info = MAGMA_ERR_DEVICE_ALLOC;
                return *info;
            }
        }
        h = 1; //num_gpus; //magma_ceildiv( n, nb );
        #ifdef USE_PINNED_CLMEMORY
        cl_mem buffer = clCreateBuffer(gContext, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(magmaDoubleComplex)*n*nb*h, NULL, NULL);
        for (d=0; d<num_gpus; d++) {
            work = (magmaDoubleComplex*)clEnqueueMapBuffer(queues[2*d], buffer, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, 
                                                           sizeof(magmaDoubleComplex)*n*nb*h, 0, NULL, NULL, NULL);
        }
        #else
        if (MAGMA_SUCCESS != magma_zmalloc_cpu( &work, n*nb*h )) {
            *info = MAGMA_ERR_HOST_ALLOC;
            return *info;
        }
        #endif
        if (uplo == MagmaUpper) {
            /* with two queues for each device */
            magma_zpotrf2_msub(num_subs, num_gpus, uplo, n, n, 0, 0, nb, d_lA, 0, ldda, 
                               dwork, lddp, work, n, h, queues, info);
            //magma_zpotrf3_msub(num_subs, num_gpus, uplo, n, n, 0, 0, nb, d_lA, 0, ldda, 
            //                   dwork, lddp, work, n, h, queues, info);
            /* with three streams */
            //magma_zpotrf3_msub(num_gpus, uplo, n, n, 0, 0, nb, d_lA, ldda, dwork, lddp, work, n,  
            //                   h, stream, event, info);
        } else {
            /* with two queues for each device */
            magma_zpotrf2_msub(num_subs, num_gpus, uplo, n, n, 0, 0, nb, d_lA, 0, ldda, 
                               dwork, lddp, work, nb*h, h, queues, info);
            //magma_zpotrf3_msub(num_subs, num_gpus, uplo, n, n, 0, 0, nb, d_lA, 0, ldda, 
            //                   dwork, lddp, work, nb*h, h, queues, info);
            //magma_zpotrf4_msub(num_subs, num_gpus, uplo, n, n, 0, 0, nb, d_lA, 0, ldda, 
            //                   dwork, lddp, work, nb*h, h, queues, info);
            /* with three streams */
            //magma_zpotrf3_msub(num_gpus, uplo, n, n, 0, 0, nb, d_lA, ldda, dwork, lddp, work, nb*h, 
            //                   h, stream, event, info);
        }

        /* clean up */
        for (d=0; d<num_gpus; d++) magma_free( dwork[d] );
        #ifdef USE_PINNED_CLMEMORY
        for (d=0; d<num_gpus; d++) {
            clEnqueueUnmapMemObject(queues[2*d], buffer, work, 0, NULL, NULL);
        }
        clReleaseMemObject( buffer );
        #else
        magma_free_cpu( work );
        #endif
    } /* end of not lapack */

    return *info;
} /* magma_zpotrf_msub */
