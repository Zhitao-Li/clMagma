/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @generated from clmagmablas/zlarfg.cl, normal z -> c, Tue Jun 18 16:14:14 2019

       auto-converted from clarfg.cu
       
       @author Mark Gates
*/
#include "kernels_header.h"
#include "clarfg.h"
#include "reduce.h"

#define COMPLEX


// ----------------------------------------
// kernel for magma_clarfg.
// Uses one block of NB (currently 512) threads.
// Each thread sums dx[ tx + k*NB ]^2 for k = 0, 1, ...,
// then does parallel sum reduction to get norm-squared.
// 
// Currently setup to use NB threads, no matter how small dx is.
__kernel void
clarfg_kernel(
    magma_int_t n,
    __global magmaFloatComplex* dalpha, unsigned long dalpha_offset,
    __global magmaFloatComplex* dx, unsigned long dx_offset, magma_int_t incx,
    __global magmaFloatComplex* dtau, unsigned long dtau_offset )
{
    dalpha += dalpha_offset;
    dx += dx_offset;
    dtau += dtau_offset;

    const int tx = get_local_id(0);
    __local float swork[ NB ];
    // TODO is it faster for each thread to have its own scale (register)?
    // if so, communicate it via swork[0]
    __local float sscale;
    __local magmaFloatComplex sscale2;
    magmaFloatComplex tmp;
    
    // find max of [dalpha, dx], to use as scaling to avoid unnecesary under- and overflow
    if ( tx == 0 ) {
        tmp = *dalpha;
        #ifdef COMPLEX
        swork[tx] = max( fabs( MAGMA_C_REAL(tmp)), fabs( MAGMA_C_IMAG(tmp)) );
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
        swork[tx] = max( swork[tx], max( fabs( MAGMA_C_REAL(tmp)), fabs( MAGMA_C_IMAG(tmp)) ));
        #else
        swork[tx] = max( swork[tx], fabs(tmp) );
        #endif
    }
    magma_smax_reduce( NB, tx, swork );
    if ( tx == 0 )
        sscale = swork[0];
    barrier( CLK_LOCAL_MEM_FENCE );
    
    // sum norm^2 of dx/sscale
    // dx has length n-1
    swork[tx] = 0;
    if ( sscale > 0 ) {
        for( int j = tx; j < n-1; j += NB ) {
            tmp = MAGMA_C_MAKE( MAGMA_C_REAL( dx[j*incx] ) / sscale,
                                MAGMA_C_IMAG( dx[j*incx] ) / sscale );
            swork[tx] += MAGMA_C_REAL(tmp)*MAGMA_C_REAL(tmp) + MAGMA_C_IMAG(tmp)*MAGMA_C_IMAG(tmp);
        }
        magma_ssum_reduce( NB, tx, swork );
    }
    
    if ( tx == 0 ) {
        magmaFloatComplex alpha = *dalpha;
        if ( swork[0] == 0
             #ifdef COMPLEX
             && MAGMA_C_IMAG(alpha) == 0
             #endif
        ) {
            // H = I
            *dtau = MAGMA_C_ZERO;
        }
        else {
            // beta = norm( [dalpha, dx] )
            float beta;
            tmp  = MAGMA_C_MAKE( MAGMA_C_REAL( alpha ) / sscale,
                                 MAGMA_C_IMAG( alpha ) / sscale );
            beta = sscale * sqrt( MAGMA_C_REAL(tmp)*MAGMA_C_REAL(tmp) + MAGMA_C_IMAG(tmp)*MAGMA_C_IMAG(tmp) + swork[0] );
            beta = -copysign( beta, MAGMA_C_REAL(alpha) );
            // todo: deal with badly scaled vectors (see lapack's larfg)
            *dtau   = MAGMA_C_MAKE( (beta - MAGMA_C_REAL(alpha)) / beta, -MAGMA_C_IMAG(alpha) / beta );
            *dalpha = MAGMA_C_MAKE( beta, 0 );
            sscale2 = MAGMA_C_DIV( MAGMA_C_ONE, MAGMA_C_SUB( alpha, MAGMA_C_MAKE( beta, 0 )));
        }
    }
    
    // scale x (if norm was not 0)
    barrier( CLK_LOCAL_MEM_FENCE );
    if ( swork[0] != 0 ) {
        for( int j = tx; j < n-1; j += NB ) {
            dx[j*incx] = MAGMA_C_MUL( dx[j*incx], sscale2 );
        }
    }
}
