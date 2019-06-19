#ifndef MAGMA_SMAX_NAN_H
#define MAGMA_SMAX_NAN_H

/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date
       
       @author Mark Gates
       @generated from clmagmablas/magma_dmax_nan.h, normal d -> s, Tue Jun 18 16:14:14 2019

       auto-converted from magma_smax_nan.cu
*/

#define NB 512

// ----------------------------------------
/// max that propogates nan consistently:
/// max_nan( 1,   nan ) = nan
/// max_nan( nan, 1   ) = nan
///
/// For x=nan, y=1:
/// nan < y is false, yields x (nan)
///
/// For x=1, y=nan:
/// x < nan    is false, would yield x, but
/// isnan(nan) is true, yields y (nan)
/* __device__ */
static inline float max_nan( float x, float y )
{
    return (isnan(y) || (x) < (y) ? (y) : (x));
}

#endif // MAGMA_SMAX_NAN_H
