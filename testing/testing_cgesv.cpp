/*
    -- clMAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @generated from testing/testing_zgesv.cpp, normal z -> c, Tue Jun 18 16:14:25 2019
       @author Mark Gates
*/
// includes, system
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// includes, project
#include "flops.h"
#include "magma.h"
#include "magma_lapack.h"
#include "testings.h"

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing cgesv
*/
int main(int argc, char **argv)
{
    TESTING_INIT();

    real_Double_t   gflops, cpu_perf, cpu_time, gpu_perf, gpu_time;
    float          error, lerror, Rnorm, Anorm, Xnorm, *work;
    magmaFloatComplex c_one     = MAGMA_C_ONE;
    magmaFloatComplex c_neg_one = MAGMA_C_NEG_ONE;
    magmaFloatComplex *h_A, *h_LU, *h_B, *h_B0, *h_X;
    magma_int_t *ipiv;
    magma_int_t N, nrhs, lda, ldb, info, sizeA, sizeB;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    magma_int_t status = 0;
    
    magma_opts opts;
    opts.parse_opts( argc, argv );
    
    float tol = opts.tolerance * lapackf77_slamch("E");
    
    nrhs = opts.nrhs;
    
    printf("%% ngpu %d\n", (int) opts.ngpu );
    if(opts.lapack){
        printf("%%   N  NRHS   CPU Gflop/s (sec)   GPU GFlop/s (sec)   ||B - AX|| / N*||A||*||X||  ||B - AX|| / N*||A||*||X||_CPU\n");
        printf("%%================================================================================================================\n");
    }else{
        printf("%%   N  NRHS   CPU Gflop/s (sec)   GPU GFlop/s (sec)   ||B - AX|| / N*||A||*||X||\n");
        printf("%%===============================================================================\n");
    }
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N = opts.nsize[itest];
            lda    = N;
            ldb    = lda;
            gflops = ( FLOPS_CGETRF( N, N ) + FLOPS_CGETRS( N, nrhs ) ) / 1e9;
            
            TESTING_MALLOC_CPU( h_A,  magmaFloatComplex, lda*N    );
            TESTING_MALLOC_CPU( h_LU, magmaFloatComplex, lda*N    );
            TESTING_MALLOC_CPU( h_B0, magmaFloatComplex, ldb*nrhs );
            TESTING_MALLOC_CPU( h_B,  magmaFloatComplex, ldb*nrhs );
            TESTING_MALLOC_CPU( h_X,  magmaFloatComplex, ldb*nrhs );
            TESTING_MALLOC_CPU( work, float,             N        );
            TESTING_MALLOC_CPU( ipiv, magma_int_t,        N        );
            
            /* Initialize the matrices */
            sizeA = lda*N;
            sizeB = ldb*nrhs;
            lapackf77_clarnv( &ione, ISEED, &sizeA, h_A );
            lapackf77_clarnv( &ione, ISEED, &sizeB, h_B );
            
            // copy A to LU and B to X; save A and B for residual
            lapackf77_clacpy( "F", &N, &N,    h_A, &lda, h_LU, &lda );
            lapackf77_clacpy( "F", &N, &nrhs, h_B, &ldb, h_X,  &ldb );
            lapackf77_clacpy( "F", &N, &nrhs, h_B, &ldb, h_B0, &ldb );
            
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            gpu_time = magma_wtime();
            magma_cgesv( N, nrhs, h_LU, lda, ipiv, h_X, ldb, opts.queues2, &info );
            gpu_time = magma_wtime() - gpu_time;
            gpu_perf = gflops / gpu_time;
            if (info != 0)
                printf("magma_cgesv returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
            
            //=====================================================================
            // Residual
            //=====================================================================
            Anorm = lapackf77_clange("I", &N, &N,    h_A, &lda, work);
            Xnorm = lapackf77_clange("I", &N, &nrhs, h_X, &ldb, work);
            
            blasf77_cgemm( MagmaNoTransStr, MagmaNoTransStr, &N, &nrhs, &N,
                           &c_one,     h_A, &lda,
                                       h_X, &ldb,
                           &c_neg_one, h_B, &ldb);
            
            Rnorm = lapackf77_clange("I", &N, &nrhs, h_B, &ldb, work);
            error = Rnorm/(N*Anorm*Xnorm);
            bool okay = (error < tol);
            status += ! okay;
            
            /* ====================================================================
               Performs operation using LAPACK
               =================================================================== */
            if ( opts.lapack ) {
                lapackf77_clacpy( "F", &N, &N,    h_A,  &lda, h_LU, &lda );
                lapackf77_clacpy( "F", &N, &nrhs, h_B0, &ldb, h_X,  &ldb );

                cpu_time = magma_wtime();
                lapackf77_cgesv( &N, &nrhs, h_LU, &lda, ipiv, h_X, &ldb, &info );
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
                if (info != 0)
                    printf("lapackf77_cgesv returned error %d: %s.\n",
                           (int) info, magma_strerror( info ));
                
                //Anorm = lapackf77_clange("I", &N, &N,    h_A, &lda, work);
                Xnorm = lapackf77_clange("I", &N, &nrhs, h_X, &ldb, work);
                blasf77_cgemm( MagmaNoTransStr, MagmaNoTransStr, &N, &nrhs, &N,
                               &c_one,     h_A, &lda,
                                           h_X, &ldb,
                               &c_neg_one, h_B0, &ldb);
                
                Rnorm = lapackf77_clange("I", &N, &nrhs, h_B0, &ldb, work);
                lerror = Rnorm/(N*Anorm*Xnorm);
                bool lokay = (lerror < tol);
                printf( "%5d %5d   %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e   %-6s           %8.2e   %s\n",
                        (int) N, (int) nrhs, cpu_perf, cpu_time, gpu_perf, gpu_time,
                        error, (okay ? "ok" : "failed"),
                        lerror, (lokay ? "ok" : "failed"));
            }
            else {
                printf( "%5d %5d     ---   (  ---  )   %7.2f (%7.2f)   %8.2e   %s\n",
                        (int) N, (int) nrhs, gpu_perf, gpu_time,
                        error, (okay ? "ok" : "failed"));
            }
            
            TESTING_FREE_CPU( h_A  );
            TESTING_FREE_CPU( h_LU );
            TESTING_FREE_CPU( h_B0 );
            TESTING_FREE_CPU( h_B  );
            TESTING_FREE_CPU( h_X  );
            TESTING_FREE_CPU( work );
            TESTING_FREE_CPU( ipiv );
            fflush( stdout );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }

    TESTING_FINALIZE();
    return status;
}
