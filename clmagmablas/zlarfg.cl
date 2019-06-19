/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c

       auto-converted from zlarfg.cu
       
       @author Mark Gates
*/
#include "kernels_header.h"
#include "zlarfg.h"
#include "reduce.h"

#define COMPLEX


// ----------------------------------------
// kernel for magma_zlarfg.
// Uses one block of NB (currently 512) threads.
// Each thread sums dx[ tx + k*NB ]^2 for k = 0, 1, ...,
// then does parallel sum reduction to get norm-squared.
// 
// Currently setup to use NB threads, no matter how small dx is.
__kernel void
zlarfg_kernel(
    magma_int_t n,
    __global magmaDoubleComplex* dalpha, unsigned long dalpha_offset,
    __global magmaDoubleComplex* dx, unsigned long dx_offset, magma_int_t incx,
    __global magmaDoubleComplex* dtau, unsigned long dtau_offset )
{
    dalpha += dalpha_offset;
    dx += dx_offset;
    dtau += dtau_offset;

    const int tx = get_local_id(0);
    __local double swork[ NB ];
    // TODO is it faster for each thread to have its own scale (register)?
    // if so, communicate it via swork[0]
    __local double sscale;
    __local magmaDoubleComplex sscale2;
    magmaDoubleComplex tmp;
    
    // find max of [dalpha, dx], to use as scaling to avoid unnecesary under- and overflow
    if ( tx == 0 ) {
        tmp = *dalpha;
        #ifdef COMPLEX
        swork[tx] = max( fabs( MAGMA_Z_REAL(tmp)), fabs( MAGMA_Z_IMAG(tmp)) );
        #else
        swork[tx] = fabs(tmp);
        #endif
    }
    else {
        swork[tx] = 0;
    }
    for( int j = tx; j < n-1; j += NB ) {
        tmp = dx[j*incx];
        #ifdef COMPLEX
        swork[tx] = max( swork[tx], max( fabs( MAGMA_Z_REAL(tmp)), fabs( MAGMA_Z_IMAG(tmp)) ));
        #else
        swork[tx] = max( swork[tx], fabs(tmp) );
        #endif
    }
    magma_dmax_reduce( NB, tx, swork );
    if ( tx == 0 )
        sscale = swork[0];
    barrier( CLK_LOCAL_MEM_FENCE );
    
    // sum norm^2 of dx/sscale
    // dx has length n-1
    swork[tx] = 0;
    if ( sscale > 0 ) {
        for( int j = tx; j < n-1; j += NB ) {
            tmp = MAGMA_Z_MAKE( MAGMA_Z_REAL( dx[j*incx] ) / sscale,
                                MAGMA_Z_IMAG( dx[j*incx] ) / sscale );
            swork[tx] += MAGMA_Z_REAL(tmp)*MAGMA_Z_REAL(tmp) + MAGMA_Z_IMAG(tmp)*MAGMA_Z_IMAG(tmp);
        }
        magma_dsum_reduce( NB, tx, swork );
    }
    
    if ( tx == 0 ) {
        magmaDoubleComplex alpha = *dalpha;
        if ( swork[0] == 0
             #ifdef COMPLEX
             && MAGMA_Z_IMAG(alpha) == 0
             #endif
        ) {
            // H = I
            *dtau = MAGMA_Z_ZERO;
        }
        else {
            // beta = norm( [dalpha, dx] )
            double beta;
            tmp  = MAGMA_Z_MAKE( MAGMA_Z_REAL( alpha ) / sscale,
                                 MAGMA_Z_IMAG( alpha ) / sscale );
            beta = sscale * sqrt( MAGMA_Z_REAL(tmp)*MAGMA_Z_REAL(tmp) + MAGMA_Z_IMAG(tmp)*MAGMA_Z_IMAG(tmp) + swork[0] );
            beta = -copysign( beta, MAGMA_Z_REAL(alpha) );
            // todo: deal with badly scaled vectors (see lapack's larfg)
            *dtau   = MAGMA_Z_MAKE( (beta - MAGMA_Z_REAL(alpha)) / beta, -MAGMA_Z_IMAG(alpha) / beta );
            *dalpha = MAGMA_Z_MAKE( beta, 0 );
            sscale2 = MAGMA_Z_DIV( MAGMA_Z_ONE, MAGMA_Z_SUB( alpha, MAGMA_Z_MAKE( beta, 0 )));
        }
    }
    
    // scale x (if norm was not 0)
    barrier( CLK_LOCAL_MEM_FENCE );
    if ( swork[0] != 0 ) {
        for( int j = tx; j < n-1; j += NB ) {
            dx[j*incx] = MAGMA_Z_MUL( dx[j*incx], sscale2 );
        }
    }
}
