/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @generated from testing/testing_zgeadd.cpp, normal z -> c, Tue Jun 18 16:14:24 2019
       @author Mark Gates
*/
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include "magma.h"
#include "magma_lapack.h"
#include "testings.h"


/* ////////////////////////////////////////////////////////////////////////////
   -- Testing cgeadd
*/
int main( int argc, char** argv)
{
    TESTING_INIT();

    real_Double_t   gflops, gpu_perf, gpu_time, cpu_perf, cpu_time;
    float          error, work[1];
    magmaFloatComplex *h_A, *h_B;
    magmaFloatComplex_ptr d_A, d_B;
    magmaFloatComplex alpha = MAGMA_C_MAKE( 3.1415, 2.718 );
    magmaFloatComplex c_neg_one = MAGMA_C_NEG_ONE;
    
    magma_int_t M, N, size, lda, ldda;
    magma_int_t ione = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    magma_int_t status = 0;

    magma_opts opts;
    opts.parse_opts( argc, argv );

    float tol = opts.tolerance * lapackf77_slamch("E");
    
    /* Uncomment these lines to check parameters.
     * magma_xerbla calls lapack's xerbla to print out error. */
    //magmablas_cgeadd( -1,  N, alpha, d_A, ldda, d_B, ldda );
    //magmablas_cgeadd(  M, -1, alpha, d_A, ldda, d_B, ldda );
    //magmablas_cgeadd(  M,  N, alpha, d_A, M-1,  d_B, ldda );
    //magmablas_cgeadd(  M,  N, alpha, d_A, ldda, d_B, N-1  );

    printf("%%   M     N   CPU GFlop/s (ms)    GPU GFlop/s (ms)    |Bl-Bm|/|Bl|\n");
    printf("%%========================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            M = opts.msize[itest];
            N = opts.nsize[itest];
            lda    = M;
            ldda   = magma_roundup( M, opts.align );  // multiple of 32 by default
            size   = lda*N;
            gflops = 2.*M*N / 1e9;
            
            TESTING_MALLOC_CPU( h_A, magmaFloatComplex, lda *N );
            TESTING_MALLOC_CPU( h_B, magmaFloatComplex, lda *N );
            
            TESTING_MALLOC_DEV( d_A, magmaFloatComplex, ldda*N );
            TESTING_MALLOC_DEV( d_B, magmaFloatComplex, ldda*N );
            
            lapackf77_clarnv( &ione, ISEED, &size, h_A );
            lapackf77_clarnv( &ione, ISEED, &size, h_B );
            
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            magma_csetmatrix( M, N, h_A, lda, d_A, 0, ldda, opts.queue );
            magma_csetmatrix( M, N, h_B, lda, d_B, 0, ldda, opts.queue );
            
            //magmablasSetKernelStream( opts.queue );
            gpu_time = magma_sync_wtime( opts.queue );
            magmablas_cgeadd( M, N, alpha, d_A, 0, ldda, d_B, 0, ldda, opts.queue );
            gpu_time = magma_sync_wtime( opts.queue ) - gpu_time;
            gpu_perf = gflops / gpu_time;
            
            /* =====================================================================
               Performs operation using LAPACK
               =================================================================== */
            cpu_time = magma_wtime();
            for( int j = 0; j < N; ++j ) {
                blasf77_caxpy( &M, &alpha, &h_A[j*lda], &ione, &h_B[j*lda], &ione );
            }
            cpu_time = magma_wtime() - cpu_time;
            cpu_perf = gflops / cpu_time;
            
            /* =====================================================================
               Check result
               =================================================================== */
            magma_cgetmatrix( M, N, d_B, 0, ldda, h_A, lda, opts.queue );
            
            error = lapackf77_clange( "F", &M, &N, h_B, &lda, work );
            blasf77_caxpy( &size, &c_neg_one, h_A, &ione, h_B, &ione );
            error = lapackf77_clange( "F", &M, &N, h_B, &lda, work ) / error;
            
            printf("%5d %5d   %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e   %s\n",
                   (int) M, (int) N,
                   cpu_perf, cpu_time*1000., gpu_perf, gpu_time*1000.,
                   error, (error < tol ? "ok" : "failed"));
            status += ! (error < tol);
            
            TESTING_FREE_CPU( h_A );
            TESTING_FREE_CPU( h_B );
            
            TESTING_FREE_DEV( d_A );
            TESTING_FREE_DEV( d_B );
            fflush( stdout );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }

    TESTING_FINALIZE();
    return status;
}