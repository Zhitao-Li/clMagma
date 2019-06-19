/*
    -- clMAGMA (version 1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @generated from clmagmablas/dznrm2.cl, normal z -> c, Tue Jun 18 16:14:14 2019
*/
#include "kernels_header.h"
#include "reduce.h"

//#define BLOCK_SIZE  512
#define BLOCK_SIZE  256
#define BLOCK_SIZEx  32
//#define BLOCK_SIZEy  16
#define BLOCK_SIZEy  8

#define PRECISION_c


//==============================================================================
__kernel void
magmablas_scnrm2_kernel( int m, __global magmaFloatComplex *da, int da_offset, int ldda, __global float *dxnorm, int dxnorm_offset )
{
    da += da_offset;
    dxnorm += dxnorm_offset;
    const int i = get_local_id(0);
    
    da += get_group_id(0) * ldda;

    __local float sum[ BLOCK_SIZE ];
    float re, lsum;

    // get norm of dx
    lsum = 0;
    for( int j = i; j < m; j += BLOCK_SIZE ) {
        #if (defined(PRECISION_s) || defined(PRECISION_d))
            re = da[j];
            lsum += re*re;
        #else
            re = MAGMA_C_REAL( da[j] );
            float im = MAGMA_C_IMAG( da[j] );
            lsum += re*re + im*im;
        #endif
    }
    sum[i] = lsum;
    magma_ssum_reduce(BLOCK_SIZE, i, sum );
    
    if (i == 0)
        dxnorm[get_group_id(0)] = sqrt(sum[0]);
}


//==============================================================================
__kernel void
magmablas_scnrm2_adjust_kernel(__global float *xnorm, int xnorm_offset, __global magmaFloatComplex *c, int c_offset)
{
    xnorm += xnorm_offset;
    c += c_offset;
     
    const int i = get_local_id(0);
    
    __local float sum[ BLOCK_SIZE ];
    float temp;
    
    temp = MAGMA_C_ABS( c[i] ) / xnorm[0];
    sum[i] = -temp * temp;
    magma_ssum_reduce( get_local_size(0), i, sum );
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    if (i == 0)
        xnorm[0] = xnorm[0] * sqrt(1+sum[0]);
}
