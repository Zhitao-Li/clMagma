/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @generated from testing/testing_zher2k.cpp, normal z -> c, Tue Jun 18 16:14:24 2019
       @author Chongxiao Cao
*/
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include "testings.h"  // before magma.h, to include cublas_v2
#include "flops.h"
#include "magma.h"
#include "magma_lapack.h"


/* ////////////////////////////////////////////////////////////////////////////
   -- Testing cher2k
*/
int main( int argc, char** argv)
{
    TESTING_INIT();

    real_Double_t   gflops, cublas_perf, cublas_time, cpu_perf, cpu_time;
    float          cublas_error, Cnorm, work[1];
    magma_int_t N, K;
    magma_int_t Ak, An, Bk, Bn;
    magma_int_t sizeA, sizeB, sizeC;
    magma_int_t lda, ldb, ldc, ldda, lddb, lddc;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    
    magmaFloatComplex *h_A, *h_B, *h_C, *h_Ccublas;
    magmaFloatComplex_ptr d_A, d_B, d_C;
    magmaFloatComplex c_neg_one = MAGMA_C_NEG_ONE;
    magmaFloatComplex alpha = MAGMA_C_MAKE(  0.29, -0.86 );
    float beta  = MAGMA_D_MAKE( -0.48,  0.38 );
    magma_int_t status = 0;
    
    magma_opts opts;
    opts.parse_opts( argc, argv );
    opts.lapack |= opts.check;  // check (-c) implies lapack (-l)
    
    float tol = opts.tolerance * lapackf77_slamch("E");
    
    printf("%% If running lapack (option --lapack), clBLAS error is computed\n"
           "%% relative to CPU BLAS result.\n\n");
    printf("%% uplo = %s, transA = %s\n",
           lapack_uplo_const(opts.uplo), lapack_trans_const(opts.transA) );
    printf("%%   N     K   clBLAS Gflop/s (ms)   CPU Gflop/s (ms)  clBLAS error\n");
    printf("%%=================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N = opts.msize[itest];
            K = opts.ksize[itest];
            gflops = FLOPS_CHER2K(K, N) / 1e9;

            if ( opts.transA == MagmaNoTrans ) {
                lda = An = N;
                Ak = K;
                ldb = Bn = N;
                Bk = K;
            } else {
                lda = An = K;
                Ak = N;
                ldb = Bn = K;
                Bk = N;
            }
            
            ldc = N;
            
            ldda = magma_roundup( lda, opts.align );  // multiple of 32 by default
            lddb = magma_roundup( ldb, opts.align );  // multiple of 32 by default
            lddc = magma_roundup( ldc, opts.align );  // multiple of 32 by default
            
            sizeA = lda*Ak;
            sizeB = ldb*Ak;
            sizeC = ldc*N;
            
            TESTING_MALLOC_CPU( h_A,       magmaFloatComplex, lda*Ak );
            TESTING_MALLOC_CPU( h_B,       magmaFloatComplex, ldb*Bk );
            TESTING_MALLOC_CPU( h_C,       magmaFloatComplex, ldc*N  );
            TESTING_MALLOC_CPU( h_Ccublas, magmaFloatComplex, ldc*N  );
            
            TESTING_MALLOC_DEV( d_A, magmaFloatComplex, ldda*Ak );
            TESTING_MALLOC_DEV( d_B, magmaFloatComplex, lddb*Bk );
            TESTING_MALLOC_DEV( d_C, magmaFloatComplex, lddc*N  );
            
            /* Initialize the matrices */
            lapackf77_clarnv( &ione, ISEED, &sizeA, h_A );
            lapackf77_clarnv( &ione, ISEED, &sizeB, h_B );
            lapackf77_clarnv( &ione, ISEED, &sizeC, h_C );
            
            /* =====================================================================
               Performs operation using clBLAS / cuBLAS
               =================================================================== */
            magma_csetmatrix( An, Ak, h_A, lda, d_A, 0, ldda, opts.queue );
            magma_csetmatrix( Bn, Bk, h_B, ldb, d_B, 0, lddb, opts.queue );
            magma_csetmatrix( N, N, h_C, ldc, d_C, 0, lddc, opts.queue );
            
            //magmablasSetKernelStream( opts.queue );  // opts.handle also uses opts.queue
            cublas_time = magma_sync_wtime( opts.queue );
            #ifdef HAVE_CUBLAS
                cublasCher2k( opts.handle, cublas_uplo_const(opts.uplo), cublas_trans_const(opts.transA), N, K,
                              &alpha, d_A, ldda,
                                      d_B, lddb,
                              &beta,  d_C, lddc );
            #else
                magma_cher2k( opts.uplo, opts.transA, N, K,
                              alpha, d_A, 0, ldda,
                                     d_B, 0, lddb,
                              beta,  d_C, 0, lddc, opts.queue );
            #endif
            cublas_time = magma_sync_wtime( opts.queue ) - cublas_time;
            cublas_perf = gflops / cublas_time;
            
            magma_cgetmatrix( N, N, d_C, 0, lddc, h_Ccublas, ldc, opts.queue );
            
            /* =====================================================================
               Performs operation using CPU BLAS
               =================================================================== */
            if ( opts.lapack ) {
                cpu_time = magma_wtime();
                blasf77_cher2k( lapack_uplo_const(opts.uplo), lapack_trans_const(opts.transA), &N, &K,
                               &alpha, h_A, &lda,
                                       h_B, &ldb,
                               &beta,  h_C, &ldc );
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
            }
            
            /* =====================================================================
               Check the result
               =================================================================== */
            if ( opts.lapack ) {
                // compute relative error for both magma & clBLAS / cuBLAS, relative to lapack,
                // |C_magma - C_lapack| / |C_lapack|
                Cnorm = lapackf77_clange( "M", &N, &N, h_C, &ldc, work );
                
                blasf77_caxpy( &sizeC, &c_neg_one, h_C, &ione, h_Ccublas, &ione );
                cublas_error = lapackf77_clange( "M", &N, &N, h_Ccublas, &ldc, work ) / Cnorm;
                
                printf("%5d %5d   %7.2f (%7.2f)   %7.2f (%7.2f)    %8.2e   %s\n",
                       (int) N, (int) K,
                       cublas_perf, 1000.*cublas_time,
                       cpu_perf,    1000.*cpu_time,
                       cublas_error, (cublas_error < tol ? "ok" : "failed"));
                status += ! (cublas_error < tol);
            }
            else {
                printf("%5d %5d   %7.2f (%7.2f)    ---   (  ---  )    ---     ---\n",
                       (int) N, (int) K,
                       cublas_perf, 1000.*cublas_time);
            }
            
            TESTING_FREE_CPU( h_A );
            TESTING_FREE_CPU( h_B );
            TESTING_FREE_CPU( h_C );
            TESTING_FREE_CPU( h_Ccublas );
            
            TESTING_FREE_DEV( d_A );
            TESTING_FREE_DEV( d_B );
            TESTING_FREE_DEV( d_C );
            fflush( stdout );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }

    TESTING_FINALIZE();
    return status;
}
