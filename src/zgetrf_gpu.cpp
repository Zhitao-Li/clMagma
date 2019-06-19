/*
    -- clMAGMA (version 0.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c

*/
#include "common_magma.h"

extern "C" magma_int_t
magma_zgetrf_gpu(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_ptr dA, size_t dA_offset, magma_int_t ldda,
    magma_int_t *ipiv,
    magma_queue_t queue,
    magma_int_t *info )
{
/*  -- clMAGMA (version 0.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

    Purpose
    =======
    ZGETRF computes an LU factorization of a general M-by-N matrix A
    using partial pivoting with row interchanges.

    The factorization has the form
       A = P * L * U
    where P is a permutation matrix, L is lower triangular with unit
    diagonal elements (lower trapezoidal if m > n), and U is upper
    triangular (upper trapezoidal if m < n).

    This is the right-looking Level 3 BLAS version of the algorithm.

    Arguments
    =========
    M       (input) INTEGER
            The number of rows of the matrix A.  M >= 0.

    N       (input) INTEGER
            The number of columns of the matrix A.  N >= 0.

    A       (input/output) COMPLEX_16 array on the GPU, dimension (LDDA,N).
            On entry, the M-by-N matrix to be factored.
            On exit, the factors L and U from the factorization
            A = P*L*U; the unit diagonal elements of L are not stored.

    LDDA     (input) INTEGER
            The leading dimension of the array A.  LDDA >= max(1,M).

    IPIV    (output) INTEGER array, dimension (min(M,N))
            The pivot indices; for 1 <= i <= min(M,N), row i of the
            matrix was interchanged with row IPIV(i).

    INFO    (output) INTEGER
            = 0:  successful exit
            < 0:  if INFO = -i, the i-th argument had an illegal value
                  or another error occured, such as memory allocation failed.
            > 0:  if INFO = i, U(i,i) is exactly zero. The factorization
                  has been completed, but the factor U is exactly
                  singular, and division by zero will occur if it is used
                  to solve a system of equations.
    =====================================================================    */

    #define  dA(i_, j_) dA,   dA_offset  + (i_)*nb       + (j_)*nb*ldda
    #define dAT(i_, j_) dAT,  dAT_offset + (i_)*nb*lddat + (j_)*nb
    #define dAP(i_, j_) dAP,               (i_)          + (j_)*maxm
    #define work(i_)   (work + (i_))

    magmaDoubleComplex c_one     = MAGMA_Z_ONE;
    magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;

    magma_int_t iinfo, nb;
    magma_int_t maxm, maxn, mindim;
    magma_int_t i, j, rows, s, lddat, ldwork;
    magmaDoubleComplex_ptr dAT, dAP;
    magmaDoubleComplex *work;
    size_t dAT_offset;

    /* Check arguments */
    *info = 0;
    if (m < 0)
        *info = -1;
    else if (n < 0)
        *info = -2;
    else if (ldda < max(1,m))
        *info = -4;

    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    /* Quick return if possible */
    if (m == 0 || n == 0)
        return *info;

    /* Function Body */
    mindim = min(m, n);
    nb     = magma_get_zgetrf_nb(m);
    s      = mindim / nb;

    if (nb <= 1 || nb >= min(m,n)) {
        /* Use CPU code. */
        if ( MAGMA_SUCCESS != magma_zmalloc_cpu(  &work, m*n )) {
          *info = MAGMA_ERR_HOST_ALLOC;
          return *info;
        }
        magma_zgetmatrix( m, n, dA(0,0), ldda, work(0), m, queue );
        lapackf77_zgetrf( &m, &n, work, &m, ipiv, info );
        magma_zsetmatrix( m, n, work(0), m, dA(0,0), ldda, queue );
        magma_free_cpu( work );
    }
    else {
        /* Use hybrid blocked code. */
        maxm = magma_roundup( m, 32 );
        maxn = magma_roundup( n, 32 );

        if ( MAGMA_SUCCESS != magma_zmalloc( &dAP, nb*maxm )) {
            *info = MAGMA_ERR_DEVICE_ALLOC;
            return *info;
        }

        // square matrices can be done in place;
        // rectangular requires copy to transpose
        if ( m == n ) {
            dAT = dA;
            dAT_offset = dA_offset;
            lddat = ldda;
            magmablas_ztranspose_inplace( m, dAT(0,0), lddat, queue );
        }
        else {
            lddat = maxn;  // N-by-M
            dAT_offset = 0;
            if ( MAGMA_SUCCESS != magma_zmalloc( &dAT, lddat*maxm )) {
                magma_free( dAP );
                *info = MAGMA_ERR_DEVICE_ALLOC;
                return *info;
            }
            magmablas_ztranspose( m, n, dA(0,0), ldda, dAT(0,0), lddat, queue );
        }

        ldwork = maxm;
        if ( MAGMA_SUCCESS != magma_zmalloc_cpu( &work, ldwork*nb )) {
            magma_free( dAP );
            if ( dA != dAT )
                magma_free( dAT );
            
            *info = MAGMA_ERR_HOST_ALLOC;
            return *info;
        }

        for( j=0; j < s; j++ ) {
            // download j-th panel
            magmablas_ztranspose( nb, m-j*nb, dAT(j,j), lddat, dAP(0,0), maxm, queue );
            magma_zgetmatrix( m-j*nb, nb, dAP(0,0), maxm, work(0), ldwork, queue );

            if ( j > 0 ){
                magma_ztrsm( MagmaRight, MagmaUpper, MagmaNoTrans, MagmaUnit,
                             n - (j+1)*nb, nb,
                             c_one, dAT(j-1,j-1), lddat,
                                    dAT(j-1,j+1), lddat, queue );
                magma_zgemm( MagmaNoTrans, MagmaNoTrans,
                             n-(j+1)*nb, m-j*nb, nb,
                             c_neg_one, dAT(j-1,j+1), lddat,
                                        dAT(j,  j-1), lddat,
                             c_one,     dAT(j,  j+1), lddat, queue );
            }

            // do the cpu part
            rows = m - j*nb;
            lapackf77_zgetrf( &rows, &nb, work, &ldwork, ipiv+j*nb, &iinfo );
            if ( *info == 0 && iinfo > 0 )
                *info = iinfo + j*nb;

            for( i=j*nb; i < j*nb + nb; ++i ) {
                ipiv[i] += j*nb;
            }
            magmablas_zlaswp( n, dAT(0,0), lddat, j*nb + 1, j*nb + nb, ipiv, 1, queue );

            // upload j-th panel
            magma_zsetmatrix( m-j*nb, nb, work(0), ldwork, dAP(0,0), maxm, queue );
            magmablas_ztranspose( m-j*nb, nb, dAP(0,0), maxm, dAT(j,j), lddat, queue );

            // do the small non-parallel computations (next panel update)
            if ( s > (j+1) ) {
                magma_ztrsm( MagmaRight, MagmaUpper, MagmaNoTrans, MagmaUnit,
                             nb, nb,
                             c_one, dAT(j, j  ), lddat,
                             dAT(j, j+1), lddat, queue );
                magma_zgemm( MagmaNoTrans, MagmaNoTrans,
                             nb, m-(j+1)*nb, nb,
                             c_neg_one, dAT(j,   j+1), lddat,
                                        dAT(j+1, j  ), lddat,
                             c_one,     dAT(j+1, j+1), lddat, queue );
            }
            else {
                magma_ztrsm( MagmaRight, MagmaUpper, MagmaNoTrans, MagmaUnit,
                             n-s*nb, nb,
                             c_one, dAT(j, j  ), lddat,
                             dAT(j, j+1), lddat, queue );
                magma_zgemm( MagmaNoTrans, MagmaNoTrans,
                             n-(j+1)*nb, m-(j+1)*nb, nb,
                             c_neg_one, dAT(j,   j+1), lddat,
                                        dAT(j+1, j  ), lddat,
                             c_one,     dAT(j+1, j+1), lddat, queue );
            }
        }

        magma_int_t nb0 = min( m - s*nb, n - s*nb );
        if ( nb0 > 0 ) {
            rows = m - s*nb;
    
            magmablas_ztranspose( nb0, rows, dAT(s,s), lddat, dAP(0,0), maxm, queue );
            magma_zgetmatrix( rows, nb0, dAP(0,0), maxm, work(0), ldwork, queue );
    
            // do the cpu part
            lapackf77_zgetrf( &rows, &nb0, work, &ldwork, ipiv+s*nb, &iinfo );
            if ( *info == 0 && iinfo > 0 )
                *info = iinfo + s*nb;
            
            for( i=s*nb; i < s*nb + nb0; ++i ) {
                ipiv[i] += s*nb;
            }
            magmablas_zlaswp( n, dAT(0,0), lddat, s*nb + 1, s*nb + nb0, ipiv, 1, queue );
    
            // upload j-th panel
            magma_zsetmatrix( rows, nb0, work(0), ldwork, dAP(0,0), maxm, queue );
            magmablas_ztranspose( rows, nb0, dAP(0,0), maxm, dAT(s,s), lddat, queue );
    
            magma_ztrsm( MagmaRight, MagmaUpper, MagmaNoTrans, MagmaUnit,
                         n-s*nb-nb0, nb0,
                         c_one, dAT(s,s),     lddat,
                                dAT(s,s)+nb0, lddat, queue );
        }

        // undo transpose
        if ( dA == dAT ) {
            magmablas_ztranspose_inplace( m, dAT(0,0), lddat, queue );
        }
        else {
            magmablas_ztranspose( n, m, dAT(0,0), lddat, dA(0,0), ldda, queue );
            magma_free( dAT );
        }

        magma_free( dAP );
        magma_free_cpu( work );
    }

    return *info;
} /* magma_zgetrf_gpu */

#undef dAT
