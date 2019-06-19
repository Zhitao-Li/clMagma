/*
 *  -- clMAGMA (version 1.1) --
 *     Univ. of Tennessee, Knoxville
 *     Univ. of California, Berkeley
 *     Univ. of Colorado, Denver
 *     @date
 *
 * @generated from testing/testing_zgemm_reduce.cpp, normal z -> s, Tue Jun 18 16:14:24 2019
 *
 **/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "flops.h"
#include "magma.h"
#include "magma_lapack.h"
#include "testings.h"


int main( int argc, char** argv)
{
    real_Double_t   gflops, magma_perf, magma_time, clblas_perf, clblas_time, cpu_perf, cpu_time;
    float      magma_error, clblas_error, work[1];
    magma_trans_t transA = MagmaNoTrans;
    magma_trans_t transB = MagmaNoTrans;

    magma_int_t istart = 1024;
    magma_int_t iend   = 6240;
    magma_int_t M, M0 = 0;
    magma_int_t N, N0 = 0;
    magma_int_t K, K0 = 0;
    magma_int_t i;
    magma_int_t Am, An, Bm, Bn;
    magma_int_t szeA, szeB, szeC;
    magma_int_t lda, ldb, ldc, ldda, lddb, lddc;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    
    float *h_A, *h_B, *h_C, *h_C2, *h_C3;
    magmaFloat_ptr d_A, d_B, d_C;
    float c_neg_one = MAGMA_S_NEG_ONE;
    float alpha = MAGMA_S_MAKE(  0.29, -0.86 );
    float beta  = MAGMA_S_MAKE( -0.48,  0.38 );
    
    int lapack = getenv("MAGMA_RUN_LAPACK") != NULL;
    int count = 1;

    printf("\nUsage: testing_sgemm [-NN|NT|TN|TT|NC|CN|TC|CT|CC] -M m -N n -K k -count c -l\n"
            "  -l  or setting $MAGMA_RUN_LAPACK runs CPU BLAS,\n"
            "      and computes both MAGMA and CLBLAS error using CPU BLAS result.\n"
            "      Else, MAGMA error is computed using CLBLAS result.\n\n");

    for( int i = 1; i < argc; ++i ) {
        if ( strcmp("-N", argv[i]) == 0 && i+1 < argc ){
            N0 = atoi(argv[++i]);
        }
        else if ( strcmp("-M", argv[i]) == 0 && i+1 < argc ){
            M0 = atoi(argv[++i]);
        }
        else if ( strcmp("-K", argv[i]) == 0 && i+1 < argc ){
            K0 = atoi(argv[++i]);
        }
        else if (strcmp("-NN", argv[i])==0){
            transA = transB = MagmaNoTrans;
        }
        else if (strcmp("-TT", argv[i])==0){
            transA = transB = MagmaTrans;
        }
        else if (strcmp("-NT", argv[i])==0){
            transA = MagmaNoTrans;
            transB = MagmaTrans;
        }
        else if (strcmp("-TN", argv[i])==0){
            transA = MagmaTrans;
            transB = MagmaNoTrans;
        }
        else if (strcmp("-NC", argv[i])==0){
            transA = MagmaNoTrans;
            transB = MagmaConjTrans;
        }
        else if (strcmp("-TC", argv[i])==0){
            transA = MagmaTrans;
            transB = MagmaConjTrans;
        }
        else if (strcmp("-CN", argv[i])==0){
            transA = MagmaConjTrans;
            transB = MagmaNoTrans;
        }
        else if (strcmp("-CT", argv[i])==0){
            transA = MagmaConjTrans;
            transB = MagmaTrans;
        }
        else if (strcmp("-CC", argv[i])==0){
            transA = transB = MagmaConjTrans;
        }
        else if (strcmp("-l", argv[i])==0) {
            lapack = true;
        }
        else if ( strcmp("-count", argv[i]) == 0 && i+1 < argc ){
            count = atoi(argv[++i]);
        }
        else {
            printf( "invalid argument: %s\n", argv[i] );
            exit(1);
        }
    }

    if ( (M0 != 0) && (N0 != 0) && (K0 != 0) )
        iend = istart + 1;
    
    M = N = K = iend;
    if ( M0 != 0 ) M = M0;
    if ( N0 != 0 ) N = N0;
    if ( K0 != 0 ) K = K0;
    
    if( transA == MagmaNoTrans ) {
        Am = M;
        An = K;
    }  else {
        Am = K;
        An = M;
    }
    
    if( transB == MagmaNoTrans ) {
        Bm = K;
        Bn = N;
    }  else {
        Bm = N;
        Bn = K;
    }
   
    /* Initialize */
    magma_queue_t  queue;
    magma_device_t device[ MagmaMaxGPUs ];
    magma_int_t num = 0;
    magma_int_t err;
    magma_init();
    err = magma_getdevices( device, MagmaMaxGPUs, &num );
    if ( err != 0 || num < 1 ) {
      fprintf( stderr, "magma_getdevices failed: %d\n", (int) err );
      exit(-1);
    }
    err = magma_queue_create( device[0], &queue );
    if ( err != 0 ) {
      fprintf( stderr, "magma_queue_create failed: %d\n", (int) err );
      exit(-1);
    }


    lda = ldc = M;
    ldb = Bm;
    
    ldda = magma_roundup( M, 32 );  // multiple of 32 by default
    lddb = magma_roundup( ldb, 32 );  // multiple of 32 by default
    lddc = ldda;

    K += 32;
    M += 32;
    N += 32;

    TESTING_MALLOC_CPU( h_A,  float, lda*K );
    TESTING_MALLOC_CPU( h_B,  float, ldb*Bn );
    TESTING_MALLOC_CPU( h_C,  float, ldc*N );
    TESTING_MALLOC_CPU( h_C2, float, ldc*N );
    TESTING_MALLOC_CPU( h_C3, float, ldc*N );

    TESTING_MALLOC_DEV( d_A, float, ldda*K );
    TESTING_MALLOC_DEV( d_B, float, lddb*Bn );
    TESTING_MALLOC_DEV( d_C, float, lddc*N );

    printf("Testing transA = %c  transB = %c\n", lapacke_trans_const(transA), lapacke_trans_const(transB));
    printf("    M     N     K   MAGMA Gflop/s (sec)  CLBLAS Gflop/s (sec)  CPU Gflop/s (sec)  MAGMA error  CLBLAS error\n");
    printf("===========================================================================================================\n");
    for( i=istart; i<iend; i = (int)(i*1.25) ) {
        for( int cnt = 0; cnt < count; ++cnt ) {
            M = N = K = i;
            if ( M0 != 0 ) M = M0;
            if ( N0 != 0 ) N = N0;
            if ( K0 != 0 ) K = K0;
    
            if( transA == MagmaNoTrans ) {
                lda = Am = M;
                An = K;
            }  else {
                lda = Am = K;
                An = M;
            }
    
            if( transB == MagmaNoTrans ) {
                ldb = Bm = K;
                Bn = N;
            }  else {
                ldb = Bm = N;
                Bn = K;
            }
            gflops = FLOPS_SGEMM( M, N, K ) / 1e9;
            ldc = M;
    
            ldda = magma_roundup( lda, 32 );  // multiple of 32 by default
            lddb = magma_roundup( ldb, 32 );  // multiple of 32 by default
            lddc = magma_roundup( ldc, 32 );  // multiple of 32 by default
    
            szeA = lda * An;
            szeB = ldb * Bn;
            szeC = ldc * N;
    
            /* Initialize the matrices */
            lapackf77_slarnv( &ione, ISEED, &szeA, h_A );
            lapackf77_slarnv( &ione, ISEED, &szeB, h_B );
            lapackf77_slarnv( &ione, ISEED, &szeC, h_C );
            
            /* =====================================================================
               Performs operation using MAGMA-BLAS
               =================================================================== */
            magma_ssetmatrix( Am, An, h_A, lda, d_A, 0, ldda, queue );
            magma_ssetmatrix( Bm, Bn, h_B, ldb, d_B, 0, lddb, queue );
            magma_ssetmatrix( M, N, h_C, ldc, d_C, 0, lddc, queue );
    
            magmablas_sgemm_reduce( M, N, K,
                    alpha, d_A, 0, ldda,
                    d_B, 0, lddb,
                    beta,  d_C, 0, lddc, queue );
            magma_ssetmatrix( M, N, h_C, ldc, d_C, 0, lddc, queue );
            magma_queue_sync(queue);
            
            magma_time = magma_wtime();
            magmablas_sgemm_reduce( M, N, K,
                    alpha, d_A, 0, ldda,
                    d_B, 0, lddb,
                    beta,  d_C, 0, lddc, queue );
            magma_queue_sync(queue);
            magma_time = magma_wtime() - magma_time;
            magma_perf = gflops / magma_time;
            
            magma_sgetmatrix( M, N, d_C, 0, lddc, h_C2, ldc, queue );
            
            /* =====================================================================
               Performs operation using CUDA-BLAS
               =================================================================== */
            magma_ssetmatrix( M, N, h_C, ldc, d_C, 0, lddc, queue );
            
            magma_sgemm( transA, transB, M, N, K,
                         alpha, d_A, 0, ldda,
                                d_B, 0, lddb,
                         beta,  d_C, 0, lddc, queue );
            magma_ssetmatrix( M, N, h_C, ldc, d_C, 0, lddc, queue );
            magma_queue_sync(queue);
            
            clblas_time = magma_wtime();
            magma_sgemm( transA, transB, M, N, K,
                         alpha, d_A, 0, ldda,
                                d_B, 0, lddb,
                         beta,  d_C, 0, lddc, queue );
            magma_queue_sync(queue);
            clblas_time = magma_wtime() - clblas_time;
            clblas_perf = gflops / clblas_time;
            
            magma_sgetmatrix( M, N, d_C, 0, lddc, h_C3, ldc, queue );
            
            /* =====================================================================
               Performs operation using BLAS
               =================================================================== */
            if ( lapack ) {
                cpu_time = magma_wtime();
                blasf77_sgemm( lapack_trans_const(transA), lapack_trans_const(transB), &M, &N, &K,
                               &alpha, h_A, &lda,
                                       h_B, &ldb,
                               &beta,  h_C, &ldc );
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
            }
            
            /* =====================================================================
               Error Computation and Performance Compariosn
               =================================================================== */
            if ( lapack ) {
                // compare both magma & clblas to lapack
                blasf77_saxpy(&szeC, &c_neg_one, h_C, &ione, h_C2, &ione);
                magma_error = lapackf77_slange("M", &M, &N, h_C2, &ldc, work);
                
                blasf77_saxpy(&szeC, &c_neg_one, h_C, &ione, h_C3, &ione);
                clblas_error = lapackf77_slange("M", &M, &N, h_C3, &ldc, work);
                
                printf("%5d %5d %5d   %7.2f (%7.4f)    %7.2f (%7.4f)   %7.2f (%7.4f)    %8.2e     %8.2e\n",
                       (int) M, (int) N, (int) K,
                       magma_perf, magma_time, clblas_perf, clblas_time, cpu_perf, cpu_time,
                       magma_error, clblas_error );
            }
            else {
                // compare magma to clblas
                blasf77_saxpy(&szeC, &c_neg_one, h_C3, &ione, h_C2, &ione);
                magma_error = lapackf77_slange("M", &M, &N, h_C2, &ldc, work);
                
                printf("%5d %5d %5d   %7.2f (%7.4f)    %7.2f (%7.4f)     ---   (  ---  )    %8.2e     ---\n",
                       (int) M, (int) N, (int) K,
                       magma_perf, magma_time, clblas_perf, clblas_time,
                       magma_error );
            }
        }
        if ( count > 1 ) {
            printf( "\n" );
        }
    }

    /* Memory clean up */
    TESTING_FREE_CPU( h_A );
    TESTING_FREE_CPU( h_B );
    TESTING_FREE_CPU( h_C );
    TESTING_FREE_CPU( h_C2 );
    TESTING_FREE_CPU( h_C3 );

    TESTING_FREE_DEV( d_A );
    TESTING_FREE_DEV( d_B );
    TESTING_FREE_DEV( d_C );

    magma_queue_destroy( queue );
    magma_finalize();
}
