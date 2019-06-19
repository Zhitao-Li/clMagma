/*
    -- clMAGMA (version 1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c
*/
#include "kernels_header.h"
#include "reduce.h"

//#define BLOCK_SIZE  512
#define BLOCK_SIZE  256
#define BLOCK_SIZEx  32
//#define BLOCK_SIZEy  16
#define BLOCK_SIZEy  8

#define PRECISION_z


//==============================================================================
__kernel void
magmablas_dznrm2_kernel( int m, __global magmaDoubleComplex *da, int da_offset, int ldda, __global double *dxnorm, int dxnorm_offset )
{
    da += da_offset;
    dxnorm += dxnorm_offset;
    const int i = get_local_id(0);
    
    da += get_group_id(0) * ldda;

    __local double sum[ BLOCK_SIZE ];
    double re, lsum;

    // get norm of dx
    lsum = 0;
    for( int j = i; j < m; j += BLOCK_SIZE ) {
        #if (defined(PRECISION_s) || defined(PRECISION_d))
            re = da[j];
            lsum += re*re;
        #else
            re = MAGMA_Z_REAL( da[j] );
            double im = MAGMA_Z_IMAG( da[j] );
            lsum += re*re + im*im;
        #endif
    }
    sum[i] = lsum;
    magma_dsum_reduce(BLOCK_SIZE, i, sum );
    
    if (i == 0)
        dxnorm[get_group_id(0)] = sqrt(sum[0]);
}


//==============================================================================
__kernel void
magmablas_dznrm2_adjust_kernel(__global double *xnorm, int xnorm_offset, __global magmaDoubleComplex *c, int c_offset)
{
    xnorm += xnorm_offset;
    c += c_offset;
     
    const int i = get_local_id(0);
    
    __local double sum[ BLOCK_SIZE ];
    double temp;
    
    temp = MAGMA_Z_ABS( c[i] ) / xnorm[0];
    sum[i] = -temp * temp;
    magma_dsum_reduce( get_local_size(0), i, sum );
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    if (i == 0)
        xnorm[0] = xnorm[0] * sqrt(1+sum[0]);
}
