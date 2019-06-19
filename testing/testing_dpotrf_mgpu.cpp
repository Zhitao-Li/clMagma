/*
    -- clMAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @generated from testing/testing_zpotrf_mgpu.cpp, normal z -> d, Tue Jun 18 16:14:25 2019
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
   -- Testing dpotrf_mgpu
*/
int main( int argc, char** argv )
{
    TESTING_INIT();

    real_Double_t    gflops, gpu_perf, gpu_time, cpu_perf=0, cpu_time=0;
    double           error, work[1];
    double c_neg_one = MAGMA_D_NEG_ONE;
    double *h_A, *h_R;
    magmaDouble_ptr d_lA[ MagmaMaxGPUs ];
    magma_int_t N, n2, lda, ldda, max_size, ngpu;
    magma_int_t info, nb;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    magma_int_t  status = 0;
    
    magma_opts opts;
    opts.parse_opts( argc, argv );
    opts.lapack |= opts.check;  // check (-c) implies lapack (-l)
    
    double tol = opts.tolerance * lapackf77_dlamch("E");
    
    /* initialize queues */
    magma_queue_t  queues[ MagmaMaxGPUs ];
    magma_queue_t  queues2[ MagmaMaxGPUs * 2 ];
    magma_device_t devices[ MagmaMaxGPUs ];
    magma_int_t err;
    err = magma_getdevices( devices, MagmaMaxGPUs, &ngpu );
    if ( err != 0 ) {
        fprintf( stderr, "magma_getdevices failed: %d\n", (int) err );
        exit(-1);
    }
    if ( ngpu < opts.ngpu ) {
        fprintf( stderr, "magma_getdevices: %d devices available, fewer than ngpu (%d)\n", ngpu, opts.ngpu );
        exit(-1);
    }
    for( int dev=0; dev < opts.ngpu; dev++ ) {
        err = magma_queue_create( devices[dev], &queues2[2*dev] );
        queues[dev] = queues2[2*dev];
        if ( err != 0 ) {
            fprintf( stderr, "magma_queue_create failed: %d\n", (int) err );
            exit(-1);
        }
        err = magma_queue_create( devices[dev], &queues2[2*dev+1] );
        if ( err != 0 ) {
            fprintf( stderr, "magma_queue_create failed: %d\n", (int) err );
            exit(-1);
        }
    }
    
    printf("%% ngpu = %d, uplo = %s\n", (int) opts.ngpu, lapack_uplo_const(opts.uplo) );
    printf("%%   N   CPU GFlop/s (sec)   GPU GFlop/s (sec)   ||R||_F / ||A||_F\n");
    printf("%%================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N = opts.nsize[itest];
            lda    = N;
            n2     = lda*N;
            nb     = magma_get_dpotrf_nb( N );
            gflops = FLOPS_DPOTRF( N ) / 1e9;
            
            // ngpu must be at least the number of blocks
            ngpu = min( opts.ngpu, magma_ceildiv(N,nb) );
            if ( ngpu < opts.ngpu ) {
                printf( " * too many GPUs for the matrix size, using %d GPUs\n", (int) ngpu );
            }
            
            // Allocate host memory for the matrix
            TESTING_MALLOC_CPU( h_A, double, n2 );
            TESTING_MALLOC_PIN( h_R, double, n2 );
            
            // Allocate device memory
            // matrix is distributed by block-rows or block-columns
            // this is maximum size that any GPU stores;
            // size is rounded up to full blocks in both rows and columns
            max_size = (1+N/(nb*ngpu))*nb * magma_roundup( N, nb );
            for( int dev=0; dev < ngpu; dev++ ) {
                //magma_setdevice( dev );
                TESTING_MALLOC_DEV( d_lA[dev], double, max_size );
            }
            
            /* Initialize the matrix */
            lapackf77_dlarnv( &ione, ISEED, &n2, h_A );
            magma_dmake_hpd( N, h_A, lda );
            lapackf77_dlacpy( MagmaUpperLowerStr, &N, &N, h_A, &lda, h_R, &lda );
            
            /* =====================================================================
               Performs operation using LAPACK
               =================================================================== */
            if ( opts.lapack ) {
                cpu_time = magma_wtime();
                lapackf77_dpotrf( lapack_uplo_const(opts.uplo), &N, h_A, &lda, &info );
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
                if (info != 0)
                    printf("lapackf77_dpotrf returned error %d: %s.\n",
                           (int) info, magma_strerror( info ));
            }
            
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            if ( opts.uplo == MagmaUpper ) {
                ldda = magma_roundup( N, nb );
                magma_dsetmatrix_1D_col_bcyclic( N, N, h_R, lda, d_lA, ldda, ngpu, nb, queues );
            }
            else {
                ldda = (1+N/(nb*ngpu))*nb;
                magma_dsetmatrix_1D_row_bcyclic( N, N, h_R, lda, d_lA, ldda, ngpu, nb, queues );
            }
            
            gpu_time = magma_wtime();
            magma_dpotrf_mgpu( ngpu, opts.uplo, N, d_lA, 0, ldda, queues2, &info );
            gpu_time = magma_wtime() - gpu_time;
            gpu_perf = gflops / gpu_time;
            if (info != 0)
                printf("magma_dpotrf_mgpu returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
            
            if ( opts.uplo == MagmaUpper ) {
                magma_dgetmatrix_1D_col_bcyclic( N, N, d_lA, ldda, h_R, lda, ngpu, nb, queues );
            }
            else {
                magma_dgetmatrix_1D_row_bcyclic( N, N, d_lA, ldda, h_R, lda, ngpu, nb, queues );
            }
            
            /* =====================================================================
               Check the result compared to LAPACK
               =================================================================== */
            for( int dev=0; dev < ngpu; dev++ ) {
                //magma_setdevice( dev );
                //magma_device_sync();
            }
            if ( opts.lapack ) {
                error = lapackf77_dlange("f", &N, &N, h_A, &lda, work );
                blasf77_daxpy( &n2, &c_neg_one, h_A, &ione, h_R, &ione );
                error = lapackf77_dlange("f", &N, &N, h_R, &lda, work ) / error;
                
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
            for( int dev=0; dev < ngpu; dev++ ) {
                //magma_setdevice( dev );
                TESTING_FREE_DEV( d_lA[dev] );
            }
            fflush( stdout );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }
    
    /* Free queues */
    for( int dev=0; dev < opts.ngpu; dev++ ) {
        magma_queue_destroy( queues2[2*dev]   );
        magma_queue_destroy( queues2[2*dev+1] );
    }
    
    TESTING_FINALIZE();
    return status;
}
