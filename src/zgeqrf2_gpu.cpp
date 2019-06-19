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

// using 2 queues, 1 for comm, 1 for comp.
extern cl_context     gContext;


extern "C" magma_int_t
magma_zgeqrf2_gpu(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_ptr dA, size_t dA_offset, magma_int_t ldda,
    magmaDoubleComplex *tau,
    magma_queue_t queues[2],
    magma_int_t *info)
{
/*  -- clMAGMA (version 0.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

    Purpose
    =======
    ZGEQRF computes a QR factorization of a complex M-by-N matrix A:
    A = Q * R.

    Arguments
    =========
    M       (input) INTEGER
            The number of rows of the matrix A.  M >= 0.

    N       (input) INTEGER
            The number of columns of the matrix A.  N >= 0.

    dA      (input/output) COMPLEX_16 array on the GPU, dimension (LDDA,N)
            On entry, the M-by-N matrix A.
            On exit, the elements on and above the diagonal of the array
            contain the min(M,N)-by-N upper trapezoidal matrix R (R is
            upper triangular if m >= n); the elements below the diagonal,
            with the array TAU, represent the orthogonal matrix Q as a
            product of min(m,n) elementary reflectors (see Further
            Details).

    LDDA    (input) INTEGER
            The leading dimension of the array dA.  LDDA >= max(1,M).
            To benefit from coalescent memory accesses LDDA must be
            divisible by 16.

    TAU     (output) COMPLEX_16 array, dimension (min(M,N))
            The scalar factors of the elementary reflectors (see Further
            Details).

    INFO    (output) INTEGER
            = 0:  successful exit
            < 0:  if INFO = -i, the i-th argument had an illegal value
                  or another error occured, such as memory allocation failed.

    Further Details
    ===============
    The matrix Q is represented as a product of elementary reflectors

        Q = H(1) H(2) . . . H(k), where k = min(m,n).

    Each H(i) has the form

        H(i) = I - tau * v * v'

    where tau is a complex scalar, and v is a complex vector with
    v(1:i-1) = 0 and v(i) = 1; v(i+1:m) is stored on exit in A(i+1:m,i),
    and tau in TAU(i).
    =====================================================================    */

    #define dA(a_1,a_2)    dA, (dA_offset + (a_1) + (a_2)*(ldda))
    #define work(a_1)      ( work + (a_1))
    #define hwork          ( work + (nb)*(m))

    magmaDoubleComplex_ptr dwork;
    magmaDoubleComplex  *work;

    magma_int_t i, k, ldwork, lddwork, old_i, old_ib, rows;
    magma_int_t nbmin, nx, ib, nb;
    magma_int_t lhwork, lwork;

    *info = 0;
    if (m < 0) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (ldda < max(1,m)) {
        *info = -4;
    }
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    k = min(m,n);
    if (k == 0)
        return *info;

    nb = magma_get_zgeqrf_nb(m);

    lwork  = (m+n) * nb;
    lhwork = lwork - (m)*nb;

    if ( MAGMA_SUCCESS != magma_zmalloc( &dwork, n*nb )) {
        *info = MAGMA_ERR_DEVICE_ALLOC;
        return *info;
    }

    /*    
    if ( MAGMA_SUCCESS != magma_zmalloc_cpu( &work, lwork ) ) {
        *info = MAGMA_ERR_HOST_ALLOC;
        magma_free( dwork );
        return *info;
    }
    */

    cl_mem buffer = clCreateBuffer(gContext, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, 
                                   sizeof(magmaDoubleComplex)*lwork, NULL, NULL);
    work = (magmaDoubleComplex*)clEnqueueMapBuffer(queues[0], buffer, CL_TRUE, 
                                                   CL_MAP_READ | CL_MAP_WRITE, 
                                                   0, lwork*sizeof(magmaDoubleComplex), 
                                                   0, NULL, NULL, NULL);

    nbmin = 2;
    nx    = nb;
    ldwork = m;
    lddwork= n;

    if (nb >= nbmin && nb < k && nx < k) {
        /* Use blocked code initially */
        old_i = 0; old_ib = nb;
        for (i = 0; i < k-nx; i += nb) {
            ib = min(k-i, nb);
            rows = m -i;
            
            magma_queue_sync( queues[1] );
            magma_zgetmatrix_async(rows, ib, dA(i, i), ldda, work(i), ldwork, queues[0], NULL);
          
            if (i > 0) {
                /* Apply H' to A(i:m,i+2*ib:n) from the left */
                magma_zlarfb_gpu( MagmaLeft, MagmaConjTrans, MagmaForward, MagmaColumnwise,
                                  m-old_i, n-old_i-2*old_ib, old_ib,
                                  dA(old_i, old_i         ), ldda, dwork,0,      lddwork,
                                  dA(old_i, old_i+2*old_ib), ldda, dwork,old_ib, lddwork, queues[1]);

                magma_zsetmatrix_async( old_ib, old_ib, work(old_i), ldwork,
                                            dA(old_i, old_i), ldda, queues[1], NULL);
            }

            magma_queue_sync( queues[0] );
            lapackf77_zgeqrf(&rows, &ib, work(i), &ldwork, tau+i, hwork, &lhwork, info);
   
            /* Form the triangular factor of the block reflector
               H = H(i) H(i+1) . . . H(i+ib-1) */
            lapackf77_zlarft( MagmaForwardStr, MagmaColumnwiseStr,
                              &rows, &ib,
                              work(i), &ldwork, tau+i, hwork, &ib);

            magma_zpanel_to_q( MagmaUpper, ib, work(i), ldwork, hwork+ib*ib );

            /* download the i-th V matrix */
            magma_zsetmatrix_async(rows, ib, work(i), ldwork, dA(i,i), ldda, queues[0], NULL);

            /* download the T matrix */
            magma_queue_sync( queues[1] );
            magma_zsetmatrix_async( ib, ib, hwork, ib, dwork, 0, lddwork, queues[0], NULL);
            magma_queue_sync( queues[0] );

            if (i + ib < n) {
                if (i+nb < k-nx) {
                    /* Apply H' to A(i:m,i+ib:i+2*ib) from the left */
                    magma_zlarfb_gpu( MagmaLeft, MagmaConjTrans, MagmaForward, MagmaColumnwise,
                                      rows, ib, ib,
                                      dA(i, i   ), ldda, dwork,0,  lddwork,
                                      dA(i, i+ib), ldda, dwork,ib, lddwork, queues[1]);
                    magma_zq_to_panel( MagmaUpper, ib, work(i), ldwork, hwork+ib*ib );
                }
                else {
                    magma_zlarfb_gpu( MagmaLeft, MagmaConjTrans, MagmaForward, MagmaColumnwise,
                                      rows, n-i-ib, ib,
                                      dA(i, i   ), ldda, dwork,0,  lddwork,
                                      dA(i, i+ib), ldda, dwork,ib, lddwork, queues[1]);
                    magma_zq_to_panel( MagmaUpper, ib, work(i), ldwork, hwork+ib*ib );
                    magma_zsetmatrix_async(ib, ib, work(i), ldwork, dA(i,i), ldda, queues[1], NULL);
                }
                old_i  = i;
                old_ib = ib;
            }
        }
    } else {
        i = 0;
    }

    magma_free(dwork);

    /* Use unblocked code to factor the last or only block. */
    if (i < k) {
        ib   = n-i;
        rows = m-i;
        magma_zgetmatrix_async(rows, ib, dA(i, i), ldda, work, rows, queues[1], NULL);
        magma_queue_sync( queues[1] );
        
        lhwork = lwork - rows*ib;
        lapackf77_zgeqrf(&rows, &ib, work, &rows, tau+i, work+ib*rows, &lhwork, info);
        
        magma_zsetmatrix_async(rows, ib, work, rows, dA(i, i), ldda, queues[1], NULL);
    }

    magma_queue_sync( queues[0] );
    magma_queue_sync( queues[1] );

    // magma_free_cpu(work);
    clEnqueueUnmapMemObject(queues[0], buffer, work, 0, NULL, NULL);
    clReleaseMemObject(buffer);

    return *info;
} /* magma_zgeqrf2_gpu */
