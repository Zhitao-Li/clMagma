/*
   -- clMAGMA (version 1.0) --
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
magma_zgeqrf_msub(
    magma_int_t num_subs, magma_int_t num_gpus, 
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_ptr *dlA, magma_int_t ldda,
    magmaDoubleComplex *tau, 
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
    ZGEQRF2_MGPU computes a QR factorization of a complex M-by-N matrix A:
    A = Q * R. This is a GPU interface of the routine.

    Arguments
    =========
    M       (input) INTEGER
            The number of rows of the matrix A.  M >= 0.

    N       (input) INTEGER
            The number of columns of the matrix A.  N >= 0.

    dA      (input/output) COMPLEX_16 array on the GPU, dimension (LDDA,N)
            On entry, the M-by-N matrix dA.
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

#define dlA(gpu,a_1,a_2) dlA[gpu], ((a_2)*(ldda) + (a_1))
#define dlA_offset(a_1, a_2) ((a_2)*(ldda) + (a_1))
#define work_ref(a_1)    ( work + (a_1))
#define hwork            ( work + (nb)*(m))

#define hwrk(a_1)        ( local_work + (a_1))
#define lhwrk            ( local_work + (nb)*(m))

    magmaDoubleComplex_ptr dwork[MagmaMaxGPUs], panel[MagmaMaxGPUs];
    size_t panel_offset[MagmaMaxGPUs];
    magmaDoubleComplex *local_work = NULL;

    magma_int_t i, j, k, ldwork, lddwork, old_i, old_ib, rows;
    magma_int_t nbmin, nx, ib, nb;
    magma_int_t lhwork, lwork;

    int panel_id = -1, i_local, n_local[MagmaMaxGPUs * MagmaMaxSubs], la_id, displacement,
        tot_subs = num_gpus * num_subs; 

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

    displacement = n * nb;
    lwork  = (m+n+64) * nb;
    lhwork = lwork - (m)*nb;

    for (i=0; i<num_gpus; i++) {
        if (MAGMA_SUCCESS != magma_zmalloc( &(dwork[i]), (n + ldda)*nb )) {
            *info = MAGMA_ERR_DEVICE_ALLOC;
            return *info;
        }
    }

    /* Set the number of local n for each GPU */
    for (i=0; i<tot_subs; i++) {
        n_local[i] = ((n/nb)/tot_subs)*nb;
        if (i < (n/nb)%tot_subs)
            n_local[i] += nb;
        else if (i == (n/nb)%tot_subs)
            n_local[i] += n%nb;
    }
    #ifdef USE_PINNED_CLMEMORY
    cl_mem buffer = clCreateBuffer(gContext, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(magmaDoubleComplex)*lwork, NULL, NULL);
    for (j=0; j<num_gpus; j++) {
        local_work = (magmaDoubleComplex*)clEnqueueMapBuffer(queues[2*j], buffer, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0,
                                                       sizeof(magmaDoubleComplex)*lwork, 0, NULL, NULL, NULL);
    }
    #else
    if (MAGMA_SUCCESS != magma_zmalloc_cpu( (&local_work), lwork )) {
        *info = -9;
        for (i=0; i<num_gpus; i++) {
            magma_free( dwork[i] );
        }
        *info = MAGMA_ERR_HOST_ALLOC;
        return *info;
    }
    #endif

    nbmin = 2;
    nx    = nb;
    ldwork = m;
    lddwork= n;

    if (nb >= nbmin && nb < k && nx < k) {
        /* Use blocked code initially */
        old_i = 0; old_ib = nb;
        for (i = 0; i < k-nx; i += nb) {
            /* Set the GPU number that holds the current panel */
            panel_id = (i/nb)%tot_subs;

            /* Set the local index where the current panel is */
            i_local = i/(nb*tot_subs)*nb;

            ib = min(k-i, nb);
            rows = m -i;
            /* Send current panel to the CPU */
            magma_queue_sync(queues[2*(panel_id%num_gpus)]);
            magma_zgetmatrix_async( rows, ib,
                                    dlA(panel_id, i, i_local), ldda,
                                    hwrk(i), ldwork, 
                                    queues[2*(panel_id%num_gpus)+1], NULL );

            if (i > 0) {
                /* Apply H' to A(i:m,i+2*ib:n) from the left; this is the look-ahead
                   application to the trailing matrix                                     */
                la_id = panel_id;

                /* only the GPU that has next panel is done look-ahead */
                magma_zlarfb_gpu( MagmaLeft, MagmaConjTrans, MagmaForward, MagmaColumnwise,
                                  m-old_i, n_local[la_id]-i_local-old_ib, old_ib,
                                  panel[la_id%num_gpus], panel_offset[la_id%num_gpus], ldda, 
                                  dwork[la_id%num_gpus], 0, lddwork,
                                  dlA(la_id, old_i, i_local+old_ib), ldda, 
                                  dwork[la_id%num_gpus], old_ib, lddwork, 
                                  queues[2*(la_id%num_gpus)]);

                la_id = ((i-nb)/nb)%tot_subs;
                magma_zsetmatrix_async( old_ib, old_ib,
                                        hwrk(old_i), ldwork,
                                        panel[la_id%num_gpus], panel_offset[la_id%num_gpus], ldda, 
                                        queues[2*(la_id%num_gpus)], NULL );
            }

            magma_queue_sync( queues[2*(panel_id%num_gpus)+1] );

            lapackf77_zgeqrf(&rows, &ib, hwrk(i), &ldwork, tau+i, lhwrk, &lhwork, info);

            // Form the triangular factor of the block reflector
            // H = H(i) H(i+1) . . . H(i+ib-1) 
            lapackf77_zlarft( MagmaForwardStr, MagmaColumnwiseStr,
                              &rows, &ib,
                              hwrk(i), &ldwork, tau+i, lhwrk, &ib);

            magma_zpanel_to_q( MagmaUpper, ib, hwrk(i), ldwork, lhwrk+ib*ib );
            // Send the current panel back to the GPUs 
            // Has to be done with asynchronous copies

            for (j=0; j<num_gpus; j++) {  
                if (j == panel_id%num_gpus){
                    panel[j] = dlA(panel_id, i, i_local);
                    panel_offset[j] = dlA_offset(i, i_local);
                } else {
                    panel[j] = dwork[j];
                    panel_offset[j] = displacement;
                }
                magma_queue_sync( queues[2*j] );
                magma_zsetmatrix_async( rows, ib,
                                        hwrk(i), ldwork,
                                        panel[j], panel_offset[j], ldda, 
                                        queues[2*j+1], NULL );

                /* Send the T matrix to the GPU. 
                   Has to be done with asynchronous copies */
                magma_zsetmatrix_async( ib, ib, lhwrk, ib,
                                        dwork[j], 0, lddwork, 
                                        queues[2*j+1], NULL );
            }

            for(j=0; j<num_gpus; j++) {
                magma_queue_sync( queues[2*j+1] );
            }

            if (i + ib < n) {
                 if (i+nb < k-nx) {
                    /* Apply H' to A(i:m,i+ib:i+2*ib) from the left;
                       This is update for the next panel; part of the look-ahead    */
                    la_id = (panel_id+1)%tot_subs;
                    int i_loc = (i+nb)/(nb*tot_subs)*nb;
                    for (j=0; j<tot_subs; j++) {
                        if (j == la_id)
                            magma_zlarfb_gpu( MagmaLeft, MagmaConjTrans, MagmaForward, MagmaColumnwise,
                                              rows, ib, ib,
                                              panel[j%num_gpus], panel_offset[j%num_gpus], ldda, 
                                              dwork[j%num_gpus], 0, lddwork,
                                              dlA(j, i, i_loc), ldda, 
                                              dwork[j%num_gpus], ib, lddwork, 
                                              queues[2*(j%num_gpus)]);
                        else if (j <= panel_id)
                            magma_zlarfb_gpu( MagmaLeft, MagmaConjTrans, MagmaForward, MagmaColumnwise,
                                              rows, n_local[j]-i_local-ib, ib,
                                              panel[j%num_gpus], panel_offset[j%num_gpus], ldda, 
                                              dwork[j%num_gpus], 0, lddwork,
                                              dlA(j, i, i_local+ib), ldda, 
                                              dwork[j%num_gpus], ib, lddwork,
                                              queues[2*(j%num_gpus)]);
                        else
                            magma_zlarfb_gpu( MagmaLeft, MagmaConjTrans, MagmaForward, MagmaColumnwise,
                                              rows, n_local[j]-i_local, ib,
                                              panel[j%num_gpus], panel_offset[j%num_gpus], ldda, 
                                              dwork[j%num_gpus], 0, lddwork,
                                              dlA(j, i, i_local), ldda, 
                                              dwork[j%num_gpus], ib, lddwork, 
                                              queues[2*(j%num_gpus)]);
                    }

                    /* Restore the panel */
                    magma_zq_to_panel( MagmaUpper, ib, hwrk(i), ldwork, lhwrk+ib*ib );
                } else {
                    /* do the entire update as we exit and there would be no lookahead */
                    la_id = (panel_id+1)%tot_subs;
                    int i_loc = (i+nb)/(nb*tot_subs)*nb;

                    magma_zlarfb_gpu( MagmaLeft, MagmaConjTrans, MagmaForward, MagmaColumnwise,
                                      rows, n_local[la_id]-i_loc, ib,
                                      panel[la_id%num_gpus], panel_offset[la_id%num_gpus], ldda, 
                                      dwork[la_id%num_gpus], 0, lddwork,
                                      dlA(la_id, i, i_loc), ldda, 
                                      dwork[la_id%num_gpus], ib, lddwork,
                                      queues[2*(la_id%num_gpus)]);
 
                    /* Restore the panel */
                    magma_zq_to_panel( MagmaUpper, ib, hwrk(i), ldwork, lhwrk+ib*ib ); 
                    
                    magma_zsetmatrix( ib, ib,
                                      hwrk(i), ldwork,
                                      dlA(panel_id, i, i_local), ldda,
                                      queues[2*(panel_id%num_gpus)]);
                }
                old_i  = i;
                old_ib = ib;
            }
        }
    } else {
        i = 0;
    }

    for (j=0; j<num_gpus; j++) {
        magma_free( dwork[j] );
    }

    /* Use unblocked code to factor the last or only block. */
    if (i < k) {
        ib   = n-i;
        rows = m-i;
        lhwork = lwork - rows*ib;

        panel_id = (panel_id+1)%tot_subs;
        int i_loc = (i)/(nb*tot_subs)*nb;

        magma_zgetmatrix( rows, ib,
                          dlA(panel_id, i, i_loc), ldda,
                          lhwrk, rows, 
                          queues[2*(panel_id%num_gpus)]);

        lhwork = lwork - rows*ib;
        lapackf77_zgeqrf(&rows, &ib, lhwrk, &rows, tau+i, lhwrk+ib*rows, &lhwork, info);

        magma_zsetmatrix( rows, ib,
                          lhwrk, rows,
                          dlA(panel_id, i, i_loc), ldda, 
                          queues[2*(panel_id%num_gpus)]);
    }
    #ifdef USE_PINNED_CLMEMORY
    #else
    magma_free_cpu( local_work );
    #endif

    return *info;
} /* magma_zgeqrf_msub */
