/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @generated from src/zgelqf.cpp, normal z -> c, Tue Jun 18 16:14:17 2019

*/
#include "common_magma.h"

/**
    Purpose
    -------
    CGELQF computes an LQ factorization of a COMPLEX M-by-N matrix A:
    A = L * Q.

    Arguments
    ---------
    @param[in]
    m       INTEGER
            The number of rows of the matrix A.  M >= 0.

    @param[in]
    n       INTEGER
            The number of columns of the matrix A.  N >= 0.

    @param[in,out]
    A       COMPLEX array, dimension (LDA,N)
            On entry, the M-by-N matrix A.
            On exit, the elements on and below the diagonal of the array
            contain the m-by-min(m,n) lower trapezoidal matrix L (L is
            lower triangular if m <= n); the elements above the diagonal,
            with the array TAU, represent the orthogonal matrix Q as a
            product of elementary reflectors (see Further Details).
    \n
            Higher performance is achieved if A is in pinned memory, e.g.
            allocated using magma_malloc_pinned.

    @param[in]
    lda     INTEGER
            The leading dimension of the array A.  LDA >= max(1,M).

    @param[out]
    tau     COMPLEX array, dimension (min(M,N))
            The scalar factors of the elementary reflectors (see Further
            Details).

    @param[out]
    work    (workspace) COMPLEX array, dimension (MAX(1,LWORK))
            On exit, if INFO = 0, WORK[0] returns the optimal LWORK.
    \n
            Higher performance is achieved if WORK is in pinned memory, e.g.
            allocated using magma_malloc_pinned.

    @param[in]
    lwork   INTEGER
            The dimension of the array WORK.  LWORK >= max(1,M).
            For optimum performance LWORK >= M*NB, where NB is the
            optimal blocksize.
    \n
            If LWORK = -1, then a workspace query is assumed; the routine
            only calculates the optimal size of the WORK array, returns
            this value as the first entry of the WORK array, and no error
            message related to LWORK is issued.

    @param[out]
    info    INTEGER
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
                  if INFO = -10 internal GPU memory allocation failed.

    Further Details
    ---------------
    The matrix Q is represented as a product of elementary reflectors

       Q = H(k) . . . H(2) H(1), where k = min(m,n).

    Each H(i) has the form

       H(i) = I - tau * v * v'

    where tau is a complex scalar, and v is a complex vector with
    v(1:i-1) = 0 and v(i) = 1; v(i+1:n) is stored on exit in A(i,i+1:n),
    and tau in TAU(i).

    @ingroup magma_cgelqf_comp
    ********************************************************************/
extern "C" magma_int_t
magma_cgelqf(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex *A,    magma_int_t lda,   magmaFloatComplex *tau,
    magmaFloatComplex *work, magma_int_t lwork,
    magma_queue_t queues[2],
    magma_int_t *info)
{
    #define  dA(i_, j_)  dA,   (dA_offset  + (i_) + (j_)*ldda)
    #define dAT(i_, j_)  dAT,  (dAT_offset + (i_) + (j_)*ldda)
    
    size_t dA_offset=0, dAT_offset;
    magmaFloatComplex_ptr dA, dAT;
    magmaFloatComplex c_one = MAGMA_C_ONE;
    magma_int_t maxm, maxn, maxdim, nb;
    magma_int_t iinfo, ldda, lddat;
    int lquery;

    /* Function Body */
    *info = 0;
    nb = magma_get_cgelqf_nb(m);

    work[0] = MAGMA_C_MAKE( (float)(m*nb), 0 );
    lquery = (lwork == -1);
    if (m < 0) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (lda < max(1,m)) {
        *info = -4;
    } else if (lwork < max(1,m) && ! lquery) {
        *info = -7;
    }
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }
    else if (lquery) {
        return *info;
    }

    /*  Quick return if possible */
    if (min(m, n) == 0) {
        work[0] = c_one;
        return *info;
    }

    maxm = magma_roundup( m, 32 );
    maxn = magma_roundup( n, 32 );
    maxdim = max(maxm, maxn);

    // copy to GPU and transpose
    if (maxdim*maxdim < 2*maxm*maxn) {
        // close to square, do everything in-place
        ldda  = maxdim;
        lddat = maxdim;

        if (MAGMA_SUCCESS != magma_cmalloc( &dA, maxdim*maxdim )) {
            *info = MAGMA_ERR_DEVICE_ALLOC;
            return *info;
        }

        magma_csetmatrix( m, n, A, lda, dA(0,0), ldda, queues[0] );
        dAT = dA;
        dAT_offset = dA_offset;
        magmablas_ctranspose_inplace( lddat, dAT(0,0), lddat, queues[0] );
    }
    else {
        // rectangular, do everything out-of-place
        ldda  = maxm;
        lddat = maxn;

        if (MAGMA_SUCCESS != magma_cmalloc( &dA, 2*maxn*maxm )) {
            *info = MAGMA_ERR_DEVICE_ALLOC;
            return *info;
        }

        magma_csetmatrix( m, n, A, lda, dA(0,0), ldda, queues[0] );

        dAT = dA;
        dAT_offset = dA_offset + maxn * maxm;
        magmablas_ctranspose( m, n, dA(0,0), ldda, dAT(0,0), lddat, queues[0] );
    }

    // factor QR
    magma_cgeqrf2_gpu(n, m, dAT(0,0), lddat, tau, queues, &iinfo);

    // undo transpose
    if (maxdim*maxdim < 2*maxm*maxn) {
        magmablas_ctranspose_inplace( lddat, dAT(0,0), lddat, queues[0] );
        magma_cgetmatrix( m, n, dA(0,0), ldda, A, lda, queues[0] );
    } else {
        magmablas_ctranspose( n, m, dAT(0,0), lddat, dA(0,0), ldda, queues[0] );
        magma_cgetmatrix( m, n, dA(0,0), ldda, A, lda, queues[0] );
    }

    magma_free( dA );

    return *info;
} /* magma_cgelqf */
