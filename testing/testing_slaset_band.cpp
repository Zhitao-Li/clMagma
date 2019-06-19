/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @generated from testing/testing_zlaset_band.cpp, normal z -> s, Tue Jun 18 16:14:24 2019
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
   -- Testing slaset_band
   Code is very similar to testing_slacpy.cpp
*/
int main( int argc, char** argv)
{
    TESTING_INIT();
    
    #define h_A(i_,j_) (h_A + (i_) + (j_)*lda)
    
    #ifdef HAVE_clBLAS
    #define d_A(i_,j_)  d_A, ((i_) + (j_)*ldda)
    #else
    #define d_A(i_,j_) (d_A + (i_) + (j_)*ldda)
    #endif

    real_Double_t    gbytes, gpu_perf, gpu_time, cpu_perf, cpu_time;
    float           error, work[1];
    float  c_neg_one = MAGMA_S_NEG_ONE;
    float *h_A, *h_R;
    magmaFloat_ptr d_A;
    float offdiag = MAGMA_S_MAKE( 1.2000, 6.7000 );
    float diag    = MAGMA_S_MAKE( 3.1415, 2.7183 );
    magma_int_t M, N, nb, cnt, size, lda, ldda;
    magma_int_t ione     = 1;
    magma_int_t status = 0;
    
    magma_opts opts;
    opts.parse_opts( argc, argv );
    
    nb = (opts.nb == 0 ? 32 : opts.nb);

    magma_uplo_t uplo[] = { MagmaLower, MagmaUpper, MagmaFull };
    
    printf("%% K = nb = %d\n", (int) nb );
    printf("%% uplo      M     N   CPU GByte/s (ms)    GPU GByte/s (ms)    check\n");
    printf("%%=================================================================\n");
    for( int iuplo = 0; iuplo < 2; ++iuplo ) {
      for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            int inset = 0;
            M = opts.msize[itest] + 2*inset;
            N = opts.nsize[itest] + 2*inset;
            lda    = M;
            ldda   = magma_roundup( M, opts.align );  // multiple of 32 by default
            size   = lda*N;
            
            TESTING_MALLOC_CPU( h_A, float, size   );
            TESTING_MALLOC_CPU( h_R, float, size   );
            
            TESTING_MALLOC_DEV( d_A, float, ldda*N );
            
            /* Initialize the matrix */
            for( int j = 0; j < N; ++j ) {
                for( int i = 0; i < M; ++i ) {
                    h_A[i + j*lda] = MAGMA_S_MAKE( i + j/10000., j );
                }
            }
            magma_ssetmatrix( M, N, h_A, lda, d_A(0,0), ldda, opts.queue );
            
            /* =====================================================================
               Performs operation on CPU
               Also count number of elements touched.
               =================================================================== */
            cpu_time = magma_wtime();
            
            cnt = 0;
            for( int j=inset; j < N-inset; ++j ) {
                for( int k=0; k < nb; ++k ) {  // set k-th sub- or super-diagonal
                    if ( k == 0 && j < M-inset ) {
                        *h_A(j,j)   = diag;
                        cnt += 1;
                    }
                    else if ( uplo[iuplo] == MagmaLower && j+k < M-inset ) {
                        *h_A(j+k,j) = offdiag;
                        cnt += 1;
                    }
                    else if ( uplo[iuplo] == MagmaUpper && j-k >= inset && j-k < M-inset ) {
                        *h_A(j-k,j) = offdiag;
                        cnt += 1;
                    }
                }
            }
            
            gbytes = cnt / 1e9;
            
            cpu_time = magma_wtime() - cpu_time;
            cpu_perf = gbytes / cpu_time;
            
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            //magmablasSetKernelStream( opts.queue );
            gpu_time = magma_sync_wtime( opts.queue );
            
            int mm = M - 2*inset;
            int nn = N - 2*inset;
            magmablas_slaset_band( uplo[iuplo], mm, nn, nb, offdiag, diag, d_A(inset,inset), ldda, opts.queue );
            
            gpu_time = magma_sync_wtime( opts.queue ) - gpu_time;
            gpu_perf = gbytes / gpu_time;
            
            /* =====================================================================
               Check the result
               =================================================================== */
            magma_sgetmatrix( M, N, d_A(0,0), ldda, h_R, lda, opts.queue );
            
            //printf( "h_R=" );  magma_sprint( M, N, h_R, lda );
            //printf( "h_A=" );  magma_sprint( M, N, h_A, lda );

            blasf77_saxpy(&size, &c_neg_one, h_A, &ione, h_R, &ione);
            error = lapackf77_slange("f", &M, &N, h_R, &lda, work);
            
            printf("%4c   %5d %5d   %7.2f (%7.2f)   %7.2f (%7.2f)   %s\n",
                   lapacke_uplo_const( uplo[iuplo] ), (int) M, (int) N,
                   cpu_perf, cpu_time*1000., gpu_perf, gpu_time*1000.,
                   (error == 0. ? "ok" : "failed") );
            status += ! (error == 0.);
            
            TESTING_FREE_CPU( h_A );
            TESTING_FREE_CPU( h_R );
            
            TESTING_FREE_DEV( d_A );
            fflush( stdout );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
      }
      printf( "\n" );
    }

    TESTING_FINALIZE();
    return status;
}
