/*
    -- clMAGMA (version 1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @generated from testing/testing_zgeqr2x_gpu.cpp, normal z -> d, Tue Jun 18 16:14:26 2019

*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include "flops.h"
#include "magma.h"
#include "magma_lapack.h"
#include "magma_operators.h"
#include "testings.h"

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing dgeqrf
*/
int main( int argc, char** argv)
{
    TESTING_INIT();

    real_Double_t    gflops, gpu_perf, gpu_time, cpu_perf, cpu_time;
    double           error, error2;

    double  c_zero    = MAGMA_D_ZERO;
    double  c_neg_one = MAGMA_D_NEG_ONE;
    double c_one     = MAGMA_D_ONE;
    double *h_A, *h_T, *h_R, *tau, *h_work, tmp[1];
    magmaDouble_ptr d_A,  d_T, ddA, dtau;
    magmaDouble_ptr d_A2, d_T2, ddA2, dtau2;
    magmaDouble_ptr dwork, dwork2;

    magma_int_t M, N, lda, ldda, lwork, n2, info, min_mn;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    magma_int_t status = 0;

    #define BLOCK_SIZE 64

    magma_opts opts;
    opts.parse_opts( argc, argv );
    
    double tol = 10. * opts.tolerance * lapackf77_dlamch("E");
    
    magma_device_t devices[MagmaMaxGPUs];
    magma_int_t ngpu, err;
    err = magma_getdevices( devices, MagmaMaxGPUs, &ngpu );
    if ( err != 0 || ngpu < 1 ) {
        fprintf( stderr, "magma_getdevices failed: %d\n", (int) err );
        exit(-1);
    }
    
    magma_queue_t queues[2];
    magma_queue_create( devices[0], &queues[0] );
    magma_queue_create( devices[0], &queues[1] );

    printf("%% version %d\n", (int) opts.version );
    printf("%% M     N     CPU GFlop/s (ms)    GPU GFlop/s (ms)   ||R - Q^H*A||   ||R_T||\n");
    printf("%%============================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            M     = opts.msize[itest];
            N     = opts.nsize[itest];

            if (N > 128) {
                printf("%5d %5d   skipping because dgeqr2x requires N <= 128\n",
                        (int) M, (int) N);
                continue;
            }
            if (M < N) {
                printf("%5d %5d   skipping because dgeqr2x requires M >= N\n",
                        (int) M, (int) N);
                continue;
            }

            min_mn = min(M, N);
            lda    = M;
            n2     = lda*N;
            ldda   = magma_roundup( M, opts.align );  // multiple of 32 by default
            gflops = (FLOPS_DGEQRF( M, N ) + FLOPS_DGEQRT( M, N )) / 1e9;

            /* Allocate memory for the matrix */
            TESTING_MALLOC_CPU( tau,   double, min_mn );
            TESTING_MALLOC_CPU( h_A,   double, n2     );
            TESTING_MALLOC_CPU( h_T,   double, N*N    );
        
            TESTING_MALLOC_PIN( h_R,   double, n2     );
        
            TESTING_MALLOC_DEV( d_A,   double, ldda*N );
            TESTING_MALLOC_DEV( d_T,   double, N*N    );
            TESTING_MALLOC_DEV( ddA,   double, N*N    );
            TESTING_MALLOC_DEV( dtau,  double, min_mn );
        
            TESTING_MALLOC_DEV( d_A2,  double, ldda*N );
            TESTING_MALLOC_DEV( d_T2,  double, N*N    );
            TESTING_MALLOC_DEV( ddA2,  double, N*N    );
            TESTING_MALLOC_DEV( dtau2, double, min_mn );
        
            TESTING_MALLOC_DEV( dwork,  double, max(5*min_mn, (BLOCK_SIZE*2+2)*min_mn) );
            TESTING_MALLOC_DEV( dwork2, double, max(5*min_mn, (BLOCK_SIZE*2+2)*min_mn) );
            
            // todo replace with magma_dlaset
            magmablas_dlaset( MagmaFull, N, N, c_zero, c_zero, ddA,  0, N, opts.queue );
            magmablas_dlaset( MagmaFull, N, N, c_zero, c_zero, d_T,  0, N, opts.queue );
            magmablas_dlaset( MagmaFull, N, N, c_zero, c_zero, ddA2, 0, N, opts.queue );
            magmablas_dlaset( MagmaFull, N, N, c_zero, c_zero, d_T2, 0, N, opts.queue );
        
            lwork = -1;
            lapackf77_dgeqrf(&M, &N, NULL, &M, NULL, tmp, &lwork, &info);
            lwork = (magma_int_t)MAGMA_D_REAL( tmp[0] );
            lwork = max(lwork, N*N);
        
            TESTING_MALLOC_CPU( h_work, double, lwork );

            /* Initialize the matrix */
            lapackf77_dlarnv( &ione, ISEED, &n2, h_A );
            lapackf77_dlacpy( MagmaUpperLowerStr, &M, &N, h_A, &lda, h_R, &lda );
            magma_dsetmatrix( M, N, h_R, lda,  d_A, 0, ldda, opts.queue );
            magma_dsetmatrix( M, N, h_R, lda, d_A2, 0, ldda, opts.queue );
    
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            gpu_time = magma_sync_wtime(0);
    
            //if (opts.version == 1)
            //    magma_dgeqr2x_gpu(M, N, d_A, ldda, dtau, d_T, ddA, dwork, &info);
            //else if (opts.version == 2)
            //    magma_dgeqr2x2_gpu(M, N, d_A, ldda, dtau, d_T, ddA, dwork, &info);
            //else if (opts.version == 3)
            
                magma_dgeqr2x3_gpu( M, N, d_A, 0, ldda, dtau, 0, d_T, 0, ddA, 0, dwork, 0, opts.queue, &info );
            
            //else {
            //    printf( "call magma_dgeqr2x4_gpu\n" );
            //    /*
            //      Going through NULL queue is faster
            //      Going through any queue is slower
            //      Doing two queues in parallel is slower than doing them sequentially
            //      Queuing happens on the NULL queue - user defined buffers are smaller?
            //    */
            //    magma_dgeqr2x4_gpu(M, N, d_A, ldda, dtau, d_T, ddA, dwork, NULL, &info);
            //    //magma_dgeqr2x4_gpu(M, N, d_A, ldda, dtau, d_T, ddA, dwork, &info, queues[1]);
            //    //magma_dgeqr2x4_gpu(M, N, d_A2, ldda, dtau2, d_T2, ddA2, dwork2, &info, queues[0]);
            //    //magma_dgeqr2x4_gpu(M, N, d_A2, ldda, dtau2, d_T2, ddA2, dwork2, &info, NULL);
            //    //gflops *= 2;
            //}
            gpu_time = magma_sync_wtime(0) - gpu_time;
            gpu_perf = gflops / gpu_time;

            if (info != 0) {
                printf("magma_dgeqr2x_gpu version %d returned error %d: %s.\n",
                       (int) opts.version, (int) info, magma_strerror( info ));
            }
            else {
                if ( opts.check ) {
                    /* =====================================================================
                       Check the result, following zqrt01 except using the reduced Q.
                       This works for any M,N (square, tall, wide).
                       =================================================================== */
                    magma_dgetmatrix( M, N, d_A, 0, ldda, h_R, M, opts.queue );
                    magma_dgetmatrix( N, N, ddA, 0, N,    h_T, N, opts.queue );
                    magma_dgetmatrix( min_mn, 1, dtau, 0, min_mn,   tau, min_mn, opts.queue );

                    // Restore the upper triangular part of A before the check
                    for (int col=0; col < N; col++) {
                        for (int row=0; row <= col; row++)
                            h_R[row + col*M] = h_T[row + col*N];
                    }

                    magma_int_t ldq = M;
                    magma_int_t ldr = min_mn;
                    double *Q, *R;
                    double *work;
                    TESTING_MALLOC_CPU( Q,    double, ldq*min_mn );  // M by K
                    TESTING_MALLOC_CPU( R,    double, ldr*N );       // K by N
                    TESTING_MALLOC_CPU( work, double,             min_mn );
                    
                    // generate M by K matrix Q, where K = min(M,N)
                    lapackf77_dlacpy( "Lower", &M, &min_mn, h_R, &M, Q, &ldq );
                    lapackf77_dorgqr( &M, &min_mn, &min_mn, Q, &ldq, tau, h_work, &lwork, &info );
                    assert( info == 0 );

                    // copy K by N matrix R
                    lapackf77_dlaset( "Lower", &min_mn, &N, &c_zero, &c_zero, R, &ldr );
                    lapackf77_dlacpy( "Upper", &min_mn, &N, h_R, &M,        R, &ldr );

                    // error = || R - Q^H*A || / (N * ||A||)
                    blasf77_dgemm( "Conj", "NoTrans", &min_mn, &N, &M,
                                   &c_neg_one, Q, &ldq, h_A, &lda, &c_one, R, &ldr );
                    double Anorm = lapackf77_dlange( "1", &M,      &N, h_A, &lda, work );
                    error2 = lapackf77_dlange( "1", &min_mn, &N, R,   &ldr, work );
                    if ( N > 0 && Anorm > 0 )
                        error2 /= (N*Anorm);

                    TESTING_FREE_CPU( Q    );  Q    = NULL;
                    TESTING_FREE_CPU( R    );  R    = NULL;
                    TESTING_FREE_CPU( work );  work = NULL;

                    /* =====================================================================
                       Performs operation using LAPACK
                       =================================================================== */
                    cpu_time = magma_wtime();
                    //lapackf77_dgeqrf(&M, &N, h_A, &lda, tau, h_work, &lwork, &info);
                    lapackf77_dlacpy( MagmaUpperLowerStr, &M, &N, h_R, &M, h_A, &lda );
                    lapackf77_dlarft( MagmaForwardStr, MagmaColumnwiseStr,
                                      &M, &N, h_A, &lda, tau, h_work, &N);
                    //magma_dgeqr2(&M, &N, h_A, &lda, tau, h_work, &info);
                    
                    cpu_time = magma_wtime() - cpu_time;
                    cpu_perf = gflops / cpu_time;
                    if (info != 0)
                        printf("lapackf77_dgeqrf returned error %d: %s.\n",
                               (int) info, magma_strerror( info ));

                    /* =====================================================================
                       Check the result compared to LAPACK
                       =================================================================== */

                    // Restore the upper triangular part of A before the check
                    for (int col=0; col < N; col++) {
                        for (int row=0; row <= col; row++)
                            h_R[row + col*M] = h_T[row + col*N];
                    }
                
                    error = lapackf77_dlange("M", &M, &N, h_A, &lda, work);
                    blasf77_daxpy(&n2, &c_neg_one, h_A, &ione, h_R, &ione);
                    error = lapackf77_dlange("M", &M, &N, h_R, &lda, work) / (N * error);
     
                    // Check if T is the same
                    magma_dgetmatrix( N, N, d_T, 0, N, h_T, N, opts.queue );
    
                    double terr = 0.;
                    for (int col=0; col < N; col++)
                        for (int row=0; row <= col; row++)
                            terr += (  MAGMA_D_ABS(h_work[row + col*N] - h_T[row + col*N])*
                                       MAGMA_D_ABS(h_work[row + col*N] - h_T[row + col*N])  );
                    terr = sqrt( terr );
    
                    // If comparison to LAPACK fail, check || R - Q^H*A || / (N * ||A||)
                    // and print fail if both fails, otherwise print ok (*)
                    printf("%5d %5d   %7.2f (%7.2f)   %7.2f (%7.2f)     %8.2e     %8.2e   %s\n",
                           (int) M, (int) N, cpu_perf, 1000.*cpu_time, gpu_perf, 1000.*gpu_time,
                           error2, terr, (error2 < tol ? "ok" : "failed" ));

                    status += ! (error2 < tol);
                }
                else {
                    printf("%5d %5d     ---   (  ---  )   %7.2f (%7.2f)     ---  \n",
                           (int) M, (int) N, gpu_perf, 1000.*gpu_time);
                }
            }
            
            TESTING_FREE_CPU( tau    );
            TESTING_FREE_CPU( h_A    );
            TESTING_FREE_CPU( h_T    );
            TESTING_FREE_CPU( h_work );
            
            TESTING_FREE_PIN( h_R    );
        
            TESTING_FREE_DEV( d_A   );
            TESTING_FREE_DEV( d_T   );
            TESTING_FREE_DEV( ddA   );
            TESTING_FREE_DEV( dtau  );
            TESTING_FREE_DEV( dwork );
        
            TESTING_FREE_DEV( d_A2   );
            TESTING_FREE_DEV( d_T2   );
            TESTING_FREE_DEV( ddA2   );
            TESTING_FREE_DEV( dtau2  );
            TESTING_FREE_DEV( dwork2 );
            fflush( stdout );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }
    
    magma_queue_destroy( queues[0] );
    magma_queue_destroy( queues[1] );

    TESTING_FINALIZE();
    return status;
}