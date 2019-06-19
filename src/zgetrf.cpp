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



extern "C" magma_int_t
magma_zgetrf(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex *A, magma_int_t lda, magma_int_t *ipiv,
    magma_queue_t queue[2],
    magma_int_t *info)
{
/*  -- clMAGMA (version 0.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

    Purpose
    =======
    ZGETRF computes an LU factorization of a general M-by-N matrix A
    using partial pivoting with row interchanges.  This version does not
    require work space on the GPU passed as input. GPU memory is allocated
    in the routine.

    The factorization has the form
       A = P * L * U
    where P is a permutation matrix, L is lower triangular with unit
    diagonal elements (lower trapezoidal if m > n), and U is upper
    triangular (upper trapezoidal if m < n).

    This is the right-looking Level 3 BLAS version of the algorithm.
    If the current stream is NULL, this version replaces it with user defined
    stream to overlap computation with communication. 

    Arguments
    =========
    M       (input) INTEGER
            The number of rows of the matrix A.  M >= 0.

    N       (input) INTEGER
            The number of columns of the matrix A.  N >= 0.

    A       (input/output) COMPLEX_16 array, dimension (LDA,N)
            On entry, the M-by-N matrix to be factored.
            On exit, the factors L and U from the factorization
            A = P*L*U; the unit diagonal elements of L are not stored.

            Higher performance is achieved if A is in pinned memory, e.g.
            allocated using magma_malloc_pinned.

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,M).

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

#define dAT(i,j) dAT, dAT_offset + ((i)*nb*lddat + (j)*nb)

    magmaDoubleComplex *work;
    magmaDoubleComplex_ptr dAT, dA, dwork, dAP;
    size_t dA_offset, dAT_offset;
    magmaDoubleComplex c_one     = MAGMA_Z_ONE;
    magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    magma_int_t     iinfo, nb;

    *info = 0;

    if (m < 0)
        *info = -1;
    else if (n < 0)
        *info = -2;
    else if (lda < max(1,m))
        *info = -4;

    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    /* Quick return if possible */
    if (m == 0 || n == 0)
        return *info;

    nb = magma_get_zgetrf_nb(m);

    if ( (nb <= 1) || (nb >= min(m,n)) ) {
        /* Use CPU code. */
        lapackf77_zgetrf(&m, &n, A, &lda, ipiv, info);
    } else {
        /* Use hybrid blocked code. */
        magma_int_t maxm, maxn, ldda, maxdim, lddat;
        magma_int_t i, j, rows, cols, s = min(m, n)/nb;
        
        maxm = magma_roundup( m, 32 );
        maxn = magma_roundup( n, 32 );

        lddat   = maxn;
        ldda    = maxm;

        maxdim = max(maxm, maxn);

        /* set number of GPUs */
        magma_int_t num_gpus = magma_num_gpus();
        if ( num_gpus > 1 ) {
            /* call multi-GPU non-GPU-resident interface  */
            printf("multiple-GPU verison not implemented\n");
            return MAGMA_ERR_NOT_IMPLEMENTED;
            // magma_zgetrf_m(num_gpus, m, n, A, lda, ipiv, info);
            // return *info;
        }

        /* explicitly checking the memory requirement */
        magma_int_t totalMem = magma_queue_meminfo( queue[0] );
        totalMem /= sizeof(magmaDoubleComplex);

        int h = 1+(2+num_gpus), num_gpus2 = num_gpus;
        int NB = (magma_int_t)(0.8*totalMem/maxm-h*nb);
        const char* ngr_nb_char = getenv("MAGMA_NGR_NB");
        if( ngr_nb_char != NULL )
            NB = max( nb, min( NB, atoi(ngr_nb_char) ) );

        if( num_gpus > ceil((double)NB/nb) ) {
            num_gpus2 = (int)ceil((double)NB/nb);
            h = 1+(2+num_gpus2);
            NB = (magma_int_t)(0.8*totalMem/maxm-h*nb);
        } 
        if( num_gpus2*NB < n ) {
            /* require too much memory, so call non-GPU-resident version */
            printf("non-GPU-resident version not implemented\n");
            return MAGMA_ERR_NOT_IMPLEMENTED; 
            //magma_zgetrf_m(num_gpus, m, n, A, lda, ipiv, info);
            //return *info;
        }

        work = A;
        if (maxdim*maxdim < 2*maxm*maxn) {
            // if close to square, allocate square matrix and transpose in-place
            if (MAGMA_SUCCESS != 
                magma_zmalloc( &dwork, (nb*maxm + maxdim*maxdim) ) ) {
                /* alloc failed so call non-GPU-resident version */
                printf("non-GPU-resident version not implemented\n");
                return MAGMA_ERR_NOT_IMPLEMENTED;
                //magma_zgetrf_m(num_gpus, m, n, A, lda, ipiv, info);
                //return *info;
            }
            dAP = dwork;

            dA = dwork;
            dA_offset = nb*maxm;            

            ldda = lddat = maxdim;
            magma_zsetmatrix( m, n, A, lda, dA, dA_offset, ldda, queue[0] );
            
            dAT = dA;
            dAT_offset = dA_offset;
            magmablas_ztranspose_inplace( m, dAT, dAT_offset, ldda, queue[0] );
        }
        else {
            // if very rectangular, allocate dA and dAT and transpose out-of-place
            if (MAGMA_SUCCESS != 
                magma_zmalloc( &dwork, (nb + maxn)*maxm )) {
                /* alloc failed so call non-GPU-resident version */
                printf("non-GPU-resident version not implemented\n");
                return MAGMA_ERR_NOT_IMPLEMENTED;
                //magma_zgetrf_m(num_gpus, m, n, A, lda, ipiv, info);
                //return *info;
            }
            dAP = dwork;

            dA = dwork;
            dA_offset = nb*maxm;
            
            magma_zsetmatrix( m, n, A, lda, dA, dA_offset, ldda, queue[0] );
            
            if (MAGMA_SUCCESS != magma_zmalloc( &dAT, maxm*maxn )) {
                /* alloc failed so call non-GPU-resident version */
                magma_free( dwork );
                printf("non-GPU-resident version not implemented\n");
                return MAGMA_ERR_NOT_IMPLEMENTED;
                //magma_zgetrf_m(num_gpus, m, n, A, lda, ipiv, info);
                //return *info;
            }
            dAT_offset = 0;   
            magmablas_ztranspose( m, n, dA, dA_offset, ldda, dAT, dAT_offset, lddat, queue[0] );
        }
        
        lapackf77_zgetrf( &m, &nb, work, &lda, ipiv, &iinfo);

        for( j = 0; j < s; j++ )
        {
            // download j-th panel
            cols = maxm - j*nb;
            
            if (j>0){
                // download j-th panel 
                magmablas_ztranspose( nb, cols, dAT(j,j), lddat, dAP, 0, cols, queue[0] );

                magma_queue_sync(queue[0]);
                magma_zgetmatrix_async( m-j*nb, nb, dAP, 0, cols, work, lda, 
                                        queue[1], NULL);
                
                magma_ztrsm( MagmaRight, MagmaUpper, MagmaNoTrans, MagmaUnit,
                             n - (j+1)*nb, nb,
                             c_one, dAT(j-1,j-1), lddat,
                                    dAT(j-1,j+1), lddat, queue[0] );
                magma_zgemm( MagmaNoTrans, MagmaNoTrans,
                             n-(j+1)*nb, m-j*nb, nb,
                             c_neg_one, dAT(j-1,j+1), lddat,
                                        dAT(j,  j-1), lddat,
                             c_one,     dAT(j,  j+1), lddat, queue[0] );

                // do the cpu part
                rows = m - j*nb;
                magma_queue_sync( queue[1] );
                lapackf77_zgetrf( &rows, &nb, work, &lda, ipiv+j*nb, &iinfo);
            }
            if (*info == 0 && iinfo > 0)
                *info = iinfo + j*nb;

            for( i=j*nb; i < j*nb + nb; ++i ) {
                ipiv[i] += j*nb;
            }
            magmablas_zlaswp( n, dAT, dAT_offset, lddat, j*nb + 1, j*nb + nb, ipiv, 1, queue[0] );

            // upload j-th panel
            magma_zsetmatrix_async( m-j*nb, nb, work, lda, dAP, 0, maxm,
                                    queue[1], NULL);
            magma_queue_sync( queue[1] );

            magmablas_ztranspose( cols, nb, dAP, 0, maxm, dAT(j,j), lddat, queue[0] );

            // do the small non-parallel computations
            if (s > (j+1)){
                magma_ztrsm( MagmaRight, MagmaUpper, MagmaNoTrans, MagmaUnit,
                             nb, nb,
                             c_one, dAT(j, j  ), lddat,
                                    dAT(j, j+1), lddat, queue[0]);
                magma_zgemm( MagmaNoTrans, MagmaNoTrans,
                             nb, m-(j+1)*nb, nb,
                             c_neg_one, dAT(j,   j+1), lddat,
                                        dAT(j+1, j  ), lddat,
                             c_one,     dAT(j+1, j+1), lddat, queue[0] );
            }
            else{
                magma_ztrsm( MagmaRight, MagmaUpper, MagmaNoTrans, MagmaUnit,
                             n-s*nb, nb,
                             c_one, dAT(j, j  ), lddat,
                                    dAT(j, j+1), lddat, queue[0] );
                magma_zgemm( MagmaNoTrans, MagmaNoTrans,
                             n-(j+1)*nb, m-(j+1)*nb, nb,
                             c_neg_one, dAT(j,   j+1), lddat,
                                        dAT(j+1, j  ), lddat,
                             c_one,     dAT(j+1, j+1), lddat, queue[0] );
            }
        }
        
        magma_int_t nb0 = min(m - s*nb, n - s*nb);
        if ( nb0 > 0 ) {
            rows = m - s*nb;
            cols = maxm - s*nb;
    
            magmablas_ztranspose( nb0, rows, dAT(s,s), lddat, dAP, 0, maxm, queue[0]);
            magma_queue_sync(queue[0]);
            magma_zgetmatrix_async( rows, nb0, dAP, 0, maxm, work, lda, queue[1], NULL );
            magma_queue_sync(queue[1]);

            // do the cpu part
            lapackf77_zgetrf( &rows, &nb0, work, &lda, ipiv+s*nb, &iinfo);
            if (*info == 0 && iinfo > 0)
                *info = iinfo + s*nb;
            
            for( i=s*nb; i < s*nb + nb0; ++i ) {
                ipiv[i] += s*nb;
            }
            magmablas_zlaswp( n, dAT, dAT_offset, lddat, s*nb + 1, s*nb + nb0, ipiv, 1, queue[0] );
    
            magma_zsetmatrix_async( rows, nb0, work, lda, dAP, 0, maxm, queue[1], NULL );
            magma_queue_sync(queue[1]);
            magmablas_ztranspose( rows, nb0, dAP, 0, maxm, dAT(s,s), lddat, queue[0]);
    
            magma_ztrsm( MagmaRight, MagmaUpper, MagmaNoTrans, MagmaUnit,
                         n-s*nb-nb0, nb0,
                         c_one, dAT(s, s),     lddat,
                                dAT(s, s)+nb0, lddat, queue[0] );
        }
       
        if (maxdim*maxdim < 2*maxm*maxn) {
            magmablas_ztranspose_inplace( m, dAT, dAT_offset, lddat, queue[0] );
            magma_zgetmatrix( m, n, dA, dA_offset, ldda, A, lda, queue[0] );
        } else {
            magmablas_ztranspose( n, m, dAT, dAT_offset, lddat, dA, dA_offset, ldda, queue[0] );
            magma_zgetmatrix( m, n, dA, dA_offset, ldda, A, lda, queue[0] );
            magma_queue_sync(queue[0]);
            magma_free( dAT );
        }

        magma_queue_sync(queue[0]);
        magma_free( dwork );
    }
    
    return *info;
} /* magma_zgetrf */

#undef dAT
