/*
 *  -- clMAGMA (version 0.1) --
 *     Univ. of Tennessee, Knoxville
 *     Univ. of California, Berkeley
 *     Univ. of Colorado, Denver
 *     @date
 *
 * @generated from testing/testing_zpotrf_msub.cpp, normal z -> s, Tue Jun 18 16:14:25 2019
 *
 **/

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

//#define USE_PINNED_CLMEMORY
#ifdef USE_PINNED_CLMEMORY
extern cl_context gContext;
#endif

#define h_A(i,j) h_A[ i + j*lda ]
void init_matrix( magma_int_t N, float *h_A, magma_int_t lda )
{
    magma_int_t ione = 1, n2 = N*lda;
    magma_int_t ISEED[4] = {0,0,0,1};
    lapackf77_slarnv( &ione, ISEED, &n2, h_A );
    /* Symmetrize and increase the diagonal */
    for (magma_int_t i = 0; i < N; ++i) {
        h_A(i,i) = MAGMA_S_MAKE( MAGMA_S_REAL(h_A(i,i)) + N, 0 );
        for (magma_int_t j = 0; j < i; ++j)
            h_A(i, j) = MAGMA_S_CNJG( h_A(j, i) );
    }
}

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing spotrf_msub
*/

int main( int argc, char** argv)
{
    real_Double_t gflops, gpu_perf, cpu_perf, gpu_time, cpu_time;
    float *h_R = NULL, *h_P = NULL;
    magmaFloat_ptr d_lA[MagmaMaxSubs * MagmaMaxGPUs];
    magma_int_t N = 0, n2, lda, ldda;
    magma_int_t size[10] =
        { 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000 };
    
    magma_int_t i, j, k, check = 0, info;
    float mz_one = MAGMA_S_NEG_ONE;
    magma_int_t ione     = 1;
   
    magma_int_t num_gpus0 = 1, num_gpus, num_subs0 = 1, num_subs, tot_subs, flag = 0;
    magma_int_t nb, n_local, nk;

    magma_uplo_t uplo = MagmaLower;

    if (argc != 1){
        for(i = 1; i<argc; i++){
            if (strcmp("-N", argv[i]) == 0){
                N = atoi(argv[++i]);
                if (N > 0) {
                    size[0] = size[9] = N;
                    flag = 1;
                }
            }
            if(strcmp("-NGPU", argv[i]) == 0)
                num_gpus0 = atoi(argv[++i]);
            if(strcmp("-NSUB", argv[i]) == 0)
                num_subs0 = atoi(argv[++i]);
            if(strcmp("-UPLO", argv[i]) == 0)
                uplo = (strcmp("L", argv[++i]) == 0 ? MagmaLower :  MagmaUpper);
            if(strcmp("-check", argv[i]) == 0)
                check = 1;
        }
    }

    /* Initialize */
    magma_queue_t  queues[2*MagmaMaxGPUs];
    magma_device_t devices[ MagmaMaxGPUs ];
    magma_int_t num = 0;
    magma_int_t err;
    magma_init();
    err = magma_getdevices( devices, MagmaMaxGPUs, &num );
    if ( err != 0 || num < 1 ) {
        fprintf( stderr, "magma_getdevices failed: %d\n", (int) err );
        exit(-1);
    }
    for(i=0;i<num_gpus0;i++){
        err = magma_queue_create( devices[i], &queues[2*i] );
        if ( err != 0 ) {
            fprintf( stderr, "magma_queue_create failed: %d\n", (int) err );
            exit(-1);
        }
        err = magma_queue_create( devices[i], &queues[2*i+1] );
        if ( err != 0 ) {
            fprintf( stderr, "magma_queue_create failed: %d\n", (int) err );
            exit(-1);
        }
    }

    printf("\nUsing %d GPUs:\n", num_gpus0);
    printf("  testing_spotrf_msub -N %d -NGPU %d -NSUB %d -UPLO %c %s\n\n", size[0], num_gpus0,num_subs0,
           (uplo == MagmaLower ? 'L' : 'U'),(check == 1 ? "-check" : " "));

    printf("  N    CPU GFlop/s (sec)    GPU GFlop/s (sec)    ||R_magma-R_lapack||_F / ||R_lapack||_F\n");
    printf("========================================================================================\n");
    for(i=0; i<10; i++){
        N   = size[i];
        lda = N;
        n2  = lda*N;
        gflops = FLOPS_SPOTRF( N ) / 1e9;;
        nb = magma_get_spotrf_nb(N);
        if (num_subs0*num_gpus0 > N/nb) {
            num_gpus = N/nb;
            num_subs = 1;
            if(N%nb != 0) num_gpus ++;
            printf("too many GPUs for the matrix size, using %d GPUs\n", (int)num_gpus);
        } else {
            num_gpus = num_gpus0;
            num_subs = num_subs0;
        }
        tot_subs = num_subs * num_gpus;
        
        /* Allocate host memory for the matrix */
        #ifdef USE_PINNED_CLMEMORY
        cl_mem buffer1 = clCreateBuffer(gContext, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, n2*sizeof(float), NULL, NULL);
        cl_mem buffer2 = clCreateBuffer(gContext, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, lda*nb*sizeof(float), NULL, NULL);
        for (k=0; k<num_gpus; k++) {
            h_R = (float*)clEnqueueMapBuffer(queues[2*k], buffer1, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, 
                                                          n2*sizeof(float), 0, NULL, NULL, NULL);
            h_P = (float*)clEnqueueMapBuffer(queues[2*k], buffer2, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, 
                                                          lda*nb*sizeof(float), 0, NULL, NULL, NULL);
        }
        #else
        TESTING_MALLOC_PIN( h_P, float, lda*nb );
        TESTING_MALLOC_PIN( h_R, float, n2     );
        #endif
        /* Initialize the matrix */
        init_matrix( N, h_R, lda );

        /* Allocate GPU memory */
        if (uplo == MagmaUpper) {
            ldda    = magma_roundup( N, nb );  // multiple of 32 by default    
            n_local = ((N+nb*tot_subs-1)/(nb*tot_subs))*nb;
        } else {
            ldda    = ((N+nb*tot_subs-1)/(nb*tot_subs))*nb;
            n_local = magma_roundup( N, nb );  // multiple of 32 by default
        }
        for (j=0; j<tot_subs; j++) {
            TESTING_MALLOC_DEV( d_lA[j], float, n_local*ldda );
        }

        /* Warm up to measure the performance */
        /* distribute matrix to gpus */
        if (uplo == MagmaUpper) {
            for (j=0; j<N; j+=nb) {
                k = (j/nb)%tot_subs;
                nk = min(nb, N-j);
                magma_ssetmatrix( j+nk, nk, 
                                 &h_R[j*lda], lda,
                                 d_lA[k], j/(nb*tot_subs)*nb*ldda, ldda, 
                                 queues[2*(k%num_gpus)]);
            }
        } else {
            for (j=0; j<N; j+=nb) {
                nk = min(nb, N-j);
                for (magma_int_t kk = 0; kk<tot_subs; kk++) {
                    magma_int_t mk = 0;
                    for (magma_int_t ii=j+kk*nb; ii<N; ii+=nb*tot_subs) {
                        magma_int_t mii = min(nb, N-ii);
                        lapackf77_slacpy( MagmaFullStr, &mii, &nk, &h_R[ii+j*lda], &lda, &h_P[mk], &lda );
                        mk += mii;
                    }
                    k = ((j+kk*nb)/nb)%tot_subs;
                    if (mk > 0 && nk > 0) {
                        magma_ssetmatrix( mk, nk, 
                                         h_P, lda,
                                         d_lA[k], j*ldda+(j+kk*nb)/(nb*tot_subs)*nb, ldda, 
                                         queues[2*(k%num_gpus)]);
                    }
                }
            }
            /*for (j=0; j<N; j+=nb) {
                k = (j/nb)%tot_subs;
                nk = min(nb, N-j);
                magma_ssetmatrix( nk, j+nk, &h_R[j], lda,
                                    d_lA[k], j/(nb*tot_subs)*nb, ldda,
                                    queues[2*(k%num_gpus)]);
            }*/
        }
        magma_spotrf_msub( num_subs, num_gpus, uplo, N, d_lA, 0, ldda, queues, &info );

        /* ====================================================================
           Performs operation using MAGMA
           =================================================================== */
        /* distribute matrix to gpus */
        if (uplo == MagmaUpper) {
            for (j=0; j<N; j+=nb) {
                k = (j/nb)%tot_subs;
                nk = min(nb, N-j);
                magma_ssetmatrix( j+nk, nk, 
                                 &h_R[j*lda], lda,
                                 d_lA[k], j/(nb*tot_subs)*nb*ldda, ldda, 
                                 queues[2*(k%num_gpus)]);
            }
        } else {
            for (j=0; j<N; j+=nb) {
                nk = min(nb, N-j);
                for (magma_int_t kk = 0; kk<tot_subs; kk++) {
                    magma_int_t mk = 0;
                    for (magma_int_t ii=j+kk*nb; ii<N; ii+=nb*tot_subs) {
                        magma_int_t mii = min(nb, N-ii);
                        lapackf77_slacpy( MagmaFullStr, &mii, &nk, &h_R[ii+j*lda], &lda, &h_P[mk], &lda );
                        mk += mii;
                    }
                    k = ((j+kk*nb)/nb)%tot_subs;
                    if (mk > 0 && nk > 0) {
                        magma_ssetmatrix( mk, nk, 
                                         h_P, lda,
                                         d_lA[k], j*ldda+(j+kk*nb)/(nb*tot_subs)*nb, ldda, 
                                         queues[2*(k%num_gpus)]);
                    }
                }
            }
            /*for (j=0; j<N; j+=nb) {
                k = (j/nb)%tot_subs;
                nk = min(nb, N-j);
                magma_ssetmatrix( nk, j+nk, &h_R[j], lda,
                                    d_lA[k], (j/(nb*tot_subs)*nb), ldda,
                                    queues[2*(k%num_gpus)]);
            }*/
        }
    
        gpu_time = magma_wtime();
        magma_spotrf_msub( num_subs, num_gpus, uplo, N, d_lA, 0, ldda, queues, &info );
        gpu_time = magma_wtime() - gpu_time;
        gpu_perf = gflops / gpu_time;
        if (info != 0)
            printf( "magma_spotrf had error %d.\n", info );
       
        /* gather matrix from gpus */
        if (uplo==MagmaUpper) {
            for (j=0; j<N; j+=nb) {
                k = (j/nb)%tot_subs;
                nk = min(nb, N-j);
                magma_sgetmatrix( j+nk, nk,
                                 d_lA[k], j/(nb*tot_subs)*nb*ldda, ldda,
                                 &h_R[j*lda], lda, queues[2*(k%num_gpus)]);
            }
        } else {
            for (j=0; j<N; j+=nb) {
                nk = min(nb, N-j);
                for (magma_int_t kk = 0; kk<tot_subs; kk++) {
                    k = ((j+kk*nb)/nb)%tot_subs;
                    magma_int_t mk = 0;
                    mk = 0;
                    for (magma_int_t ii=j+kk*nb; ii<N; ii+=nb*tot_subs) {
                        mk += min(nb, N-ii);
                    }
                    if (mk > 0 && nk > 0) {
                        magma_sgetmatrix( mk, nk, 
                                         d_lA[k], j*ldda+(j+kk*nb)/(nb*tot_subs)*nb, ldda, 
                                         h_P, lda,
                                         queues[2*(k%num_gpus)]);
                    }
                    mk = 0;
                    for (magma_int_t ii=j+kk*nb; ii<N; ii+=nb*tot_subs) {
                        magma_int_t mii = min(nb, N-ii);
                        lapackf77_slacpy( MagmaFullStr, &mii, &nk, &h_P[mk], &lda, &h_R[ii+j*lda], &lda );
                        mk += mii;
                    }
                }
            }
            /*for (j=0; j<N; j+=nb) {
                k = (j/nb)%tot_subs;
                nk = min(nb, N-j);
                magma_sgetmatrix( nk, j+nk, 
                            d_lA[k], (j/(nb*tot_subs)*nb), ldda, 
                            &h_R[j], lda, queues[2*(k%num_gpus)] );
            }*/
        }

        /* =====================================================================
           Performs operation using LAPACK
           =================================================================== */
        if (check == 1) {
            float work[1], matnorm, diffnorm;
            float *h_A;
            TESTING_MALLOC_PIN( h_A, float, n2 );
            init_matrix( N, h_A, lda );

            cpu_time = magma_wtime();
            if (uplo == MagmaLower) {
                lapackf77_spotrf( MagmaLowerStr, &N, h_A, &lda, &info );
            } else {
                lapackf77_spotrf( MagmaUpperStr, &N, h_A, &lda, &info );
            }
            cpu_time = magma_wtime() - cpu_time;
            cpu_perf = gflops / cpu_time;
            if (info != 0)
                printf( "lapackf77_spotrf had error %d.\n", info );
        
            /* =====================================================================
               Check the result compared to LAPACK
               |R_magma - R_lapack| / |R_lapack|
               =================================================================== */
            matnorm = lapackf77_slange("f", &N, &N, h_A, &lda, work);
            blasf77_saxpy(&n2, &mz_one, h_A, &ione, h_R, &ione);
            diffnorm = lapackf77_slange("f", &N, &N, h_R, &lda, work);
            printf( "%5d     %6.2f (%6.2f)     %6.2f (%6.2f)         %e\n",
                    N, cpu_perf, cpu_time, gpu_perf, gpu_time, diffnorm / matnorm );
        
            TESTING_FREE_PIN( h_A );
        } else {
            printf( "%5d      - -     (- -)     %6.2f (%6.2f)          - -\n",
                    N, gpu_perf, gpu_time );
        }
        // free memory
        #ifdef USE_PINNED_CLMEMORY
        for (k=0; k<num_gpus; k++) {
            clEnqueueUnmapMemObject(queues[2*k], buffer1, h_R, 0, NULL, NULL);
            clEnqueueUnmapMemObject(queues[2*k], buffer2, h_P, 0, NULL, NULL);
        }
        clReleaseMemObject(buffer1);
        clReleaseMemObject(buffer2);
        #else
        TESTING_FREE_PIN( h_P );
        TESTING_FREE_PIN( h_R );
        #endif
        for (j=0; j<tot_subs; j++) {
            TESTING_FREE_DEV( d_lA[j] );
        }
        if (flag != 0)
            break;
    }

    /* clean up */
    for (i=0; i<num_gpus; i++) {
        magma_queue_destroy( queues[2*i] );
        magma_queue_destroy( queues[2*i+1] );
    }
    magma_finalize();
    return 0;
}
