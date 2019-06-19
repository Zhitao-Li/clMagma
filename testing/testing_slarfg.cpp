/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @generated from testing/testing_zlarfg.cpp, normal z -> s, Tue Jun 18 16:14:24 2019
       @author Mark Gates
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "flops.h"
#include "magma.h"
#include "magma_lapack.h"
#include "magma_operators.h"
#include "testings.h"

#define PRECISION_d

int main( int argc, char** argv)
{
    TESTING_INIT();

    real_Double_t   gflops, gpu_perf, gpu_time, cpu_perf, cpu_time;
    float *h_x, *h_x2, *h_tau, *h_tau2, *h_appl, *h_work;
    float tmp;
    magmaFloat_ptr d_x, d_tau;
    float c_neg_one = MAGMA_S_NEG_ONE;
    float      error, error_tau, error_appl, work[1];
    magma_int_t N, nb, lda, ldda, size;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};    magma_int_t status = 0;

    magma_opts opts;
    opts.parse_opts( argc, argv );

    float tol = opts.tolerance * lapackf77_slamch("E");
    
    // does larfg on nb columns, one after another
    nb = max( 4, (opts.nb > 0 ? opts.nb : 64));
    
    printf("%%   N    nb    CPU GFLop/s (ms)    GPU GFlop/s (ms)   error      tau error   H*x error\n");
    printf("%%=====================================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N = opts.nsize[itest];
            lda  = N;
            ldda = magma_roundup( N, opts.align );  // multiple of 32 by default
            gflops = FLOPS_SLARFG( N ) / 1e9 * nb;
    
            TESTING_MALLOC_CPU( h_x,    float, lda*nb );
            TESTING_MALLOC_CPU( h_x2,   float, lda*nb );
            TESTING_MALLOC_CPU( h_appl, float, lda*nb );
            TESTING_MALLOC_CPU( h_tau,  float, nb   );
            TESTING_MALLOC_CPU( h_tau2, float, nb   );
            TESTING_MALLOC_CPU( h_work, float, N    );
        
            TESTING_MALLOC_DEV( d_x,   float, ldda*nb );
            TESTING_MALLOC_DEV( d_tau, float, nb      );
            
            /* Initialize the vectors */
            size = N*nb;
            lapackf77_slarnv( &ione, ISEED, &size, h_x );
            
            // try 4 special cases, 
            assert( nb >= 4 );
            const float c_zero = MAGMA_S_ZERO;
            const float a1 = MAGMA_S_MAKE( -0.1,  0   );  // real
            const float a2 = MAGMA_S_MAKE(  0,   -0.1 );  // imag
            const float a3 = MAGMA_S_MAKE( -0.1, -0.1 );  // real
            lapackf77_slaset( "full", &N, &ione, &c_zero, &c_zero, &h_x[0],     &lda );  // set column 1 to zero
            lapackf77_slaset( "full", &N, &ione, &c_zero, &a1,     &h_x[1*lda], &lda );  // set column 2 to [ real,    0, 0, ... ]
            lapackf77_slaset( "full", &N, &ione, &c_zero, &a2,     &h_x[2*lda], &lda );  // set column 3 to [ imag,    0, 0, ... ]
            lapackf77_slaset( "full", &N, &ione, &c_zero, &a3,     &h_x[3*lda], &lda );  // set column 3 to [ real, 0, 0, ... ]
            
            lapackf77_slacpy( "full", &N, &nb, h_x, &lda, h_appl, &lda );
            
            /* =====================================================================
               Performs operation using MAGMABLAS
               =================================================================== */
            magma_ssetmatrix( N, nb, h_x, lda, d_x, 0, ldda, opts.queue );
            
            //magmablasSetKernelStream( opts.queue );
            gpu_time = magma_sync_wtime( opts.queue );
            for( int j = 0; j < nb; ++j ) {
                // TODO d_x(i,j), etc. macros
                magmablas_slarfg( N, d_x, j*ldda, d_x, 1+j*ldda, ione, d_tau, j, opts.queue );
            }
            gpu_time = magma_sync_wtime( opts.queue ) - gpu_time;
            gpu_perf = gflops / gpu_time;
            
            magma_sgetmatrix( N, nb, d_x,   0, ldda, h_x2,   lda, opts.queue );
            magma_sgetvector( nb,    d_tau, 0, 1,    h_tau2, 1,   opts.queue );
            
            // check error applying Householder (larf) to original vectors
            error_appl = 0;
            for( int j = 0; j < nb; ++j ) {
                // V must have explicit 1 at top (oddly, since larfg makes 1 implicit)
                // slarfg does H^H*x, but slarf does H*x, so conjugate tau to get H^H
                tmp = h_x2[j*lda];
                h_x2[j*lda] = MAGMA_S_ONE;
                h_tau2[j] = conj( h_tau2[j] ); 
                
                lapackf77_slarf( "left", &N, &ione, &h_x2[j*lda], &ione, &h_tau2[j], &h_appl[j*lda], &lda, h_work );
                
                // restore
                h_tau2[j] = conj( h_tau2[j] );
                h_x2[j*lda] = tmp;
                
                error_appl += MAGMA_S_ABS( MAGMA_S_SUB( h_appl[j*lda], h_x2[j*lda] ));  // first elements should be equal
                error_appl += magma_cblas_snrm2( N-1, &h_appl[1+j*lda], ione );        // rest should be zero
            }
            error_appl /= (N*nb);
            
            if ( opts.verbose ) {
                printf( "X="     );  magma_sprint( N, nb, h_x,    lda );
                printf( "H^H*X=" );  magma_sprint( N, nb, h_appl, lda );
            }
            
            /* =====================================================================
               Performs operation using LAPACK
               =================================================================== */
            cpu_time = magma_wtime();
            for( int j = 0; j < nb; ++j ) {
                lapackf77_slarfg( &N, &h_x[0+j*lda], &h_x[1+j*lda], &ione, &h_tau[j] );
            }
            cpu_time = magma_wtime() - cpu_time;
            cpu_perf = gflops / cpu_time;
            
            /* =====================================================================
               Error Computation and Performance Comparison
               =================================================================== */
            blasf77_saxpy( &size, &c_neg_one, h_x, &ione, h_x2, &ione );
            error = lapackf77_slange( "F", &N, &nb, h_x2, &lda, work )
                  / lapackf77_slange( "F", &N, &nb, h_x,  &lda, work );
            
            // tau can be 0
            blasf77_saxpy( &nb, &c_neg_one, h_tau, &ione, h_tau2, &ione );
            error_tau = lapackf77_slange( "F", &nb, &ione, h_tau,  &nb, work );
            if ( error_tau != 0 ) {
                error_tau = lapackf77_slange( "F", &nb, &ione, h_tau2, &nb, work ) / error_tau;
            }

            bool okay = (error < tol && error_tau < tol && error_appl < tol);
            status += ! okay;
            printf("%5d %5d   %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e   %8.2e    %8.2e   %s\n",
                   (int) N, (int) nb, cpu_perf, 1000.*cpu_time, gpu_perf, 1000.*gpu_time,
                   error, error_tau, error_appl,
                   (okay ? "ok" : "failed") );
            
            TESTING_FREE_CPU( h_x    );
            TESTING_FREE_CPU( h_x2   );
            TESTING_FREE_CPU( h_appl );
            TESTING_FREE_CPU( h_tau  );
            TESTING_FREE_CPU( h_tau2 );
            TESTING_FREE_CPU( h_work );
        
            TESTING_FREE_DEV( d_x   );
            TESTING_FREE_DEV( d_tau );
            fflush( stdout );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }

    TESTING_FINALIZE();
    return status;
}
