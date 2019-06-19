/*
    -- clMAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date
  
       @generated from testing/testing_ztrtri_gpu.cpp, normal z -> d, Tue Jun 18 16:14:25 2019
       
       @author Mark Gates
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
#include "testings.h"

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing dtrtri
*/
int main( int argc, char** argv)
{
    TESTING_INIT();

    real_Double_t   gflops, gpu_perf, gpu_time, cpu_perf, cpu_time;
    double *h_A, *h_R;
    magmaDouble_ptr d_A;
    double c_neg_one = MAGMA_D_NEG_ONE;
    magma_int_t N, n2, lda, ldda, info;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    double      work[1], error, norm;
    magma_int_t status = 0;

    magma_opts opts;
    opts.parse_opts( argc, argv );
    opts.lapack |= opts.check;  // check (-c) implies lapack (-l)
    
    double tol = opts.tolerance * lapackf77_dlamch("E");
    
    printf("%% uplo = %s\n", lapack_uplo_const(opts.uplo) );
    printf("%%   N   CPU GFlop/s (sec)   GPU GFlop/s (sec)   ||R||_F / ||A||_F\n");
    printf("%%================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N = opts.nsize[itest];
            lda    = N;
            n2     = lda*N;
            ldda   = magma_roundup( N, opts.align );  // multiple of 32 by default
            gflops = FLOPS_DTRTRI( N ) / 1e9;
            
            TESTING_MALLOC_CPU( h_A, double, n2 );
            TESTING_MALLOC_PIN( h_R, double, n2 );
            TESTING_MALLOC_DEV( d_A, double, ldda*N );
            
            /* Initialize the matrix */
            lapackf77_dlarnv( &ione, ISEED, &n2, h_A );
            magma_dmake_hpd( N, h_A, lda );
            lapackf77_dlacpy( MagmaUpperLowerStr, &N, &N, h_A, &lda, h_R, &lda );
            
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            /* factorize matrix */
            lapackf77_dpotrf( lapack_uplo_const(opts.uplo), &N, h_A, &lda, &info );
            magma_dsetmatrix( N, N, h_A, lda, d_A, 0, ldda, opts.queue );
            //magma_dpotrf_gpu( opts.uplo, N, d_A, 0, ldda, opts.queue, &info );
            
            // check for exact singularity
            //magma_dgetmatrix( N, N, d_A, 0, ldda, h_R, lda, opts.queue );
            //h_R[ 10 + 10*lda ] = MAGMA_D_MAKE( 0.0, 0.0 );
            //magma_dsetmatrix( N, N, h_R, lda, d_A, 0, ldda, opts.queue );
            
            gpu_time = magma_wtime();
            magma_dtrtri_gpu( opts.uplo, opts.diag, N, d_A, 0, ldda, opts.queues2, &info );
            gpu_time = magma_wtime() - gpu_time;
            gpu_perf = gflops / gpu_time;
            if (info != 0)
                printf("magma_dtrtri_gpu returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
            
            /* =====================================================================
               Performs operation using LAPACK
               =================================================================== */
            if ( opts.lapack ) {
                // done above
                //lapackf77_dpotrf( lapack_uplo_const(opts.uplo), &N, h_A, &lda, &info );
                
                cpu_time = magma_wtime();
                lapackf77_dtrtri( lapack_uplo_const(opts.uplo), lapack_diag_const(opts.diag), &N, h_A, &lda, &info );
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
                if (info != 0)
                    printf("lapackf77_dtrtri returned error %d: %s.\n",
                           (int) info, magma_strerror( info ));
                
                /* =====================================================================
                   Check the result compared to LAPACK
                   =================================================================== */
                magma_dgetmatrix( N, N, d_A, 0, ldda, h_R, lda, opts.queue );
                if ( opts.verbose ) {
                    printf( "A=" );  magma_dprint( N, N, h_A, lda );
                    printf( "R=" );  magma_dprint( N, N, h_R, lda );
                }
                norm  = lapackf77_dlantr("f", lapack_uplo_const(opts.uplo), MagmaNonUnitStr, &N, &N, h_A, &lda, work);
                blasf77_daxpy(&n2, &c_neg_one, h_A, &ione, h_R, &ione);
                error = lapackf77_dlantr("f", lapack_uplo_const(opts.uplo), MagmaNonUnitStr, &N, &N, h_R, &lda, work) / norm;
                if ( opts.verbose ) {
                    printf( "diff=" );  magma_dprint( N, N, h_R, lda );
                }
                printf("%5d   %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e   %s\n",
                       (int) N, cpu_perf, cpu_time, gpu_perf, gpu_time,
                       error, (error < tol ? "ok" : "failed") );
                status += ! (error < tol);
            }
            else {
                printf("%5d     ---   (  ---  )   %7.2f (%7.2f)     ---\n",
                       (int) N, gpu_perf, gpu_time );
            }
            
            TESTING_FREE_CPU( h_A );
            TESTING_FREE_PIN( h_R );
            TESTING_FREE_DEV( d_A );
            fflush( stdout );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }

    TESTING_FINALIZE();
    return status;
}
