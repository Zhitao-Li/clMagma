/*
    -- clMAGMA (version 0.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date
                                                                                                              
       @generated from src/zpotrs_gpu.cpp, normal z -> s, Tue Jun 18 16:14:15 2019
*/
#include "common_magma.h"


extern "C" magma_int_t
magma_spotrs_gpu(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs,
    magmaFloat_ptr dA, size_t dA_offset, magma_int_t ldda,
    magmaFloat_ptr dB, size_t dB_offset, magma_int_t lddb,
    magma_queue_t queue,
    magma_int_t *info )
{
/*  -- clMagma (version 0.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date
 
    Purpose
    =======
    SPOTRS solves a system of linear equations A*X = B with a symmetric
    positive definite matrix A using the Cholesky factorization
    A = U**H*U or A = L*L**H computed by SPOTRF.

    Arguments
    =========
    UPLO    (input) CHARACTER*1
            = 'U':  Upper triangle of A is stored;
            = 'L':  Lower triangle of A is stored.

    N       (input) INTEGER
            The order of the matrix A.  N >= 0.

    NRHS    (input) INTEGER
            The number of right hand sides, i.e., the number of columns
            of the matrix B.  NRHS >= 0.

    dA      (input) REAL array on the GPU, dimension (LDDA,N)
            The triangular factor U or L from the Cholesky factorization
            A = U**H*U or A = L*L**H, as computed by SPOTRF.

    LDDA    (input) INTEGER
            The leading dimension of the array A.  LDDA >= max(1,N).

    dB      (input/output) REAL array on the GPU, dimension (LDDB,NRHS)
            On entry, the right hand side matrix B.
            On exit, the solution matrix X.

    LDDB    (input) INTEGER
            The leading dimension of the array B.  LDDB >= max(1,N).

    INFO    (output) INTEGER
            = 0:  successful exit
            < 0:  if INFO = -i, the i-th argument had an illegal value
    =====================================================================   */

    float z_one = MAGMA_S_MAKE(  1.0, 0.0 );
    
    *info = 0 ;
    if( (uplo != MagmaUpper) && (uplo != MagmaLower) )
        *info = -1;
    if( n < 0 )
        *info = -2;
    if( nrhs < 0)
        *info = -3;
    if ( ldda < max(1, n) )
        *info = -5;
    if ( lddb < max(1, n) )
        *info = -7;
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    /* Quick return if possible */
    if ( (n == 0) || (nrhs == 0) ) {
        return *info;
    }

    if( uplo== MagmaUpper){
        if ( nrhs == 1) {
            magma_strsv(MagmaUpper, MagmaConjTrans, MagmaNonUnit, n, dA, dA_offset, ldda, dB, dB_offset, 1, queue);
            magma_strsv(MagmaUpper, MagmaNoTrans,   MagmaNonUnit, n, dA, dA_offset, ldda, dB, dB_offset, 1, queue);
        } else {
            magma_strsm(MagmaLeft, MagmaUpper, MagmaConjTrans, MagmaNonUnit, n, nrhs, z_one, dA, dA_offset, ldda, dB, dB_offset, lddb, queue);
            magma_strsm(MagmaLeft, MagmaUpper, MagmaNoTrans,   MagmaNonUnit, n, nrhs, z_one, dA, dA_offset, ldda, dB, dB_offset, lddb, queue);
        }
    }
    else{
        if ( nrhs == 1) {
            magma_strsv(MagmaLower, MagmaNoTrans,   MagmaNonUnit, n, dA, dA_offset, ldda, dB, dB_offset, 1, queue );
            magma_strsv(MagmaLower, MagmaConjTrans, MagmaNonUnit, n, dA, dA_offset, ldda, dB, dB_offset, 1, queue );
        } else {
            magma_strsm(MagmaLeft, MagmaLower, MagmaNoTrans,   MagmaNonUnit, n, nrhs, z_one, dA, dA_offset, ldda, dB, dB_offset, lddb, queue);
            magma_strsm(MagmaLeft, MagmaLower, MagmaConjTrans, MagmaNonUnit, n, nrhs, z_one, dA, dA_offset, ldda, dB, dB_offset, lddb, queue);
        }
    }
    magma_queue_sync( queue );
    return *info;
}
