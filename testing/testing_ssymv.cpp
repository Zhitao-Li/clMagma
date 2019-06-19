/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @generated from testing/testing_zhemv.cpp, normal z -> s, Tue Jun 18 16:14:24 2019
       
       @author Mark Gates
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

#define PRECISION_s

int main(int argc, char **argv)
{
    TESTING_INIT();

    const float c_neg_one = MAGMA_S_NEG_ONE;
    const magma_int_t        ione      = 1;
    
    real_Double_t   atomics_perf=0, atomics_time=0;
    real_Double_t   gflops, magma_perf=0, magma_time=0, cublas_perf, cublas_time, cpu_perf, cpu_time;
    float          magma_error=0, atomics_error=0, cublas_error, work[1];
    magma_int_t ISEED[4] = {0,0,0,1};
    magma_int_t N, lda, ldda, sizeA, sizeX, sizeY, blocks, ldwork;
    magma_int_t incx = 1;
    magma_int_t incy = 1;
    magma_int_t nb   = 64;
    float alpha = MAGMA_S_MAKE(  1.5, -2.3 );
    float beta  = MAGMA_S_MAKE( -0.6,  0.8 );
    float *A, *X, *Y, *Yatomics, *Ycublas, *Ymagma;
    magmaFloat_ptr dA, dX, dY, dwork;
    magma_int_t status = 0;
    
    magma_opts opts;
    opts.parse_opts( argc, argv );
    
    float tol = opts.tolerance * lapackf77_slamch("E");

    printf("%% uplo = %s\n", lapack_uplo_const(opts.uplo) );
    printf("%%   N   MAGMA Gflop/s (ms)    Atomics Gflop/s      clBLAS Gflop/s       CPU Gflop/s   MAGMA error  Atomics    clBLAS\n");
    printf("%%=====================================================================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N = opts.nsize[itest];
            lda    = N;
            ldda   = magma_roundup( N, opts.align );  // multiple of 32 by default
            sizeA  = N*lda;
            sizeX  = N*incx;
            sizeY  = N*incy;
            gflops = FLOPS_SSYMV( N ) / 1e9;
            
            TESTING_MALLOC_CPU( A,        float, sizeA );
            TESTING_MALLOC_CPU( X,        float, sizeX );
            TESTING_MALLOC_CPU( Y,        float, sizeY );
            TESTING_MALLOC_CPU( Yatomics, float, sizeY );
            TESTING_MALLOC_CPU( Ycublas,  float, sizeY );
            TESTING_MALLOC_CPU( Ymagma,   float, sizeY );
            
            TESTING_MALLOC_DEV( dA, float, ldda*N );
            TESTING_MALLOC_DEV( dX, float, sizeX );
            TESTING_MALLOC_DEV( dY, float, sizeY );
            
            blocks = magma_ceildiv( N, nb );
            ldwork = ldda*blocks;
            TESTING_MALLOC_DEV( dwork, float, ldwork );
            
            magmablas_slaset( MagmaFull, ldwork, 1, MAGMA_S_NAN, MAGMA_S_NAN, dwork, 0, ldwork, opts.queue );
            magmablas_slaset( MagmaFull, ldda,   N, MAGMA_S_NAN, MAGMA_S_NAN, dA,    0, ldda,   opts.queue );
            
            /* Initialize the matrix */
            lapackf77_slarnv( &ione, ISEED, &sizeA, A );
            magma_smake_symmetric( N, A, lda );
            
            // should not use data from the opposite triangle -- fill with NAN to check
            magma_int_t N1 = N-1;
            if ( opts.uplo == MagmaUpper ) {
                lapackf77_slaset( "Lower", &N1, &N1, &MAGMA_S_NAN, &MAGMA_S_NAN, &A[1], &lda );
            }
            else {
                lapackf77_slaset( "Upper", &N1, &N1, &MAGMA_S_NAN, &MAGMA_S_NAN, &A[lda], &lda );
            }
            
            lapackf77_slarnv( &ione, ISEED, &sizeX, X );
            lapackf77_slarnv( &ione, ISEED, &sizeY, Y );
            
            /* =====================================================================
               Performs operation using clBLAS / cuBLAS
               =================================================================== */
            magma_ssetmatrix( N, N, A, lda, dA, 0, ldda, opts.queue );
            magma_ssetvector( N, X, incx, dX, 0, incx, opts.queue );
            magma_ssetvector( N, Y, incy, dY, 0, incy, opts.queue );
            
            //magmablasSetKernelStream( opts.queue );  // opts.handle also uses opts.queue
            cublas_time = magma_sync_wtime( opts.queue );
            #ifdef HAVE_CUBLAS
                cublasSsymv( opts.handle, cublas_uplo_const(opts.uplo),
                             N, &alpha, dA, ldda, dX, incx, &beta, dY, incy );
            #else
                magma_ssymv( opts.uplo, N, alpha, dA, 0, ldda, dX, 0, incx, beta, dY, 0, incy, opts.queue );
            #endif
            cublas_time = magma_sync_wtime( opts.queue ) - cublas_time;
            cublas_perf = gflops / cublas_time;
            
            magma_sgetvector( N, dY, 0, incy, Ycublas, incy, opts.queue );
            
            /* =====================================================================
               Performs operation using cuBLAS - using atomics
               =================================================================== */
            #ifdef HAVE_CUBLAS
                cublasSetAtomicsMode( opts.handle, CUBLAS_ATOMICS_ALLOWED );
                magma_ssetvector( N, Y, incy, dY, 0, incy, opts.queue );
                
                atomics_time = magma_sync_wtime( opts.queue );
                cublasSsymv( opts.handle, cublas_uplo_const(opts.uplo),
                             N, &alpha, dA, ldda, dX, incx, &beta, dY, incy );
                atomics_time = magma_sync_wtime( opts.queue ) - atomics_time;
                atomics_perf = gflops / atomics_time;
                
                magma_sgetvector( N, dY, 0, incy, Yatomics, incy, opts.queue );
                cublasSetAtomicsMode( opts.handle, CUBLAS_ATOMICS_NOT_ALLOWED );
            #endif
            
            /* =====================================================================
               Performs operation using MAGMABLAS (currently only in CUDA)
               =================================================================== */
            #ifdef HAVE_CUBLAS
                magma_ssetvector( N, Y, incy, dY, 0, incy, opts.queue );
                
                magma_time = magma_sync_wtime( opts.queue );
                if ( opts.version == 1 ) {
                    magmablas_ssymv_work( opts.uplo, N, alpha, dA, ldda, dX, incx, beta, dY, incy, dwork, ldwork, opts.queue );
                }
                else {
                    // non-work interface (has added overhead)
                    magmablas_ssymv( opts.uplo, N, alpha, dA, ldda, dX, incx, beta, dY, incy );
                }
                magma_time = magma_sync_wtime( opts.queue ) - magma_time;
                magma_perf = gflops / magma_time;
                
                magma_sgetvector( N, dY, 0, incy, Ymagma, incy, opts.queue );
            #endif
            
            /* =====================================================================
               Performs operation using CPU BLAS
               =================================================================== */
            cpu_time = magma_wtime();
            blasf77_ssymv( lapack_uplo_const(opts.uplo), &N, &alpha, A, &lda, X, &incx, &beta, Y, &incy );
            cpu_time = magma_wtime() - cpu_time;
            cpu_perf = gflops / cpu_time;
            
            /* =====================================================================
               Check the result
               =================================================================== */
            blasf77_saxpy( &N, &c_neg_one, Y, &incy, Ycublas, &incy );
            cublas_error = lapackf77_slange( "M", &N, &ione, Ycublas, &N, work ) / N;
            
            #ifdef HAVE_CUBLAS
                blasf77_saxpy( &N, &c_neg_one, Y, &incy, Yatomics, &incy );
                atomics_error = lapackf77_slange( "M", &N, &ione, Yatomics, &N, work ) / N;
                
                blasf77_saxpy( &N, &c_neg_one, Y, &incy, Ymagma, &incy );
                magma_error = lapackf77_slange( "M", &N, &ione, Ymagma, &N, work ) / N;
            #endif
            
            bool okay = (magma_error < tol && cublas_error < tol && atomics_error < tol);
            status += ! okay;
            printf("%5d   %7.2f (%7.2f)   %7.2f (%7.2f)   %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e   %8.2e   %8.2e   %s\n",
                   (int) N,
                   magma_perf,   1000.*magma_time,
                   atomics_perf, 1000.*atomics_time,
                   cublas_perf,  1000.*cublas_time,
                   cpu_perf,     1000.*cpu_time,
                   magma_error, cublas_error, atomics_error,
                   (okay ? "ok" : "failed"));
            
            TESTING_FREE_CPU( A );
            TESTING_FREE_CPU( X );
            TESTING_FREE_CPU( Y );
            TESTING_FREE_CPU( Ycublas  );
            TESTING_FREE_CPU( Yatomics );
            TESTING_FREE_CPU( Ymagma   );
            
            TESTING_FREE_DEV( dA );
            TESTING_FREE_DEV( dX );
            TESTING_FREE_DEV( dY );
            TESTING_FREE_DEV( dwork );
            fflush( stdout );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
      }

    TESTING_FINALIZE();
    return status;
}
