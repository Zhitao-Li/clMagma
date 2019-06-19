/*
    -- clMAGMA (version 1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @generated from clmagmablas/zlarfbx.cl, normal z -> c, Tue Jun 18 16:14:14 2019

*/
#include "kernels_header.h"
#include "reduce.h"

//#define BLOCK_SIZE 768
#define BLOCK_SIZE 256


//==============================================================================
__kernel void
magma_cgemv_kernel1(int m, __global magmaFloatComplex *V, int V_offset, int ldv,
                    __global magmaFloatComplex *c, int c_offset,
                    __global magmaFloatComplex *dwork, int dwork_offset)
{
    V += V_offset;
    c += c_offset;
    dwork += dwork_offset;
    
    const int i = get_local_id(0);
    //const magmaFloatComplex *dV = V + (get_group_id(0)) * ldv;
    V += (get_group_id(0)) * ldv;
    
    __local magmaFloatComplex sum[ BLOCK_SIZE ];
    magmaFloatComplex lsum;
    
    /*  lsum := v' * C  */
    lsum = MAGMA_C_ZERO;
    for( int j = i; j < m; j += BLOCK_SIZE )
        lsum += MAGMA_C_MUL( MAGMA_C_CNJG( V[j] ), c[j] );
    
    sum[i] = lsum;
    magma_csum_reduce( BLOCK_SIZE, i, sum );
    
    barrier(CLK_LOCAL_MEM_FENCE);
    if (i == 0)
        dwork [get_group_id(0)] = sum[0];
}


//==============================================================================
__kernel void
magma_cgemv_kernel2(int m, int n, __global magmaFloatComplex *V, int V_offset, int ldv,
                    __global magmaFloatComplex *x, int x_offset,
                    __global magmaFloatComplex *c, int c_offset)
{
    V += V_offset;
    x += x_offset;
    c += c_offset;
    
    const int i = get_local_id(0);
    const int j = i + BLOCK_SIZE * get_group_id(0);
    magmaFloatComplex lsum;
    
    V += j;
    
    lsum = MAGMA_C_ZERO;
    if (j < m) {
        for(int k=0; k<n; k++)
            lsum += MAGMA_C_MUL( V[k*ldv], x[k]);
        
        c[j] -= lsum;
    }
}


//==============================================================================
__kernel void
magma_cgemv_kernel3(int m, __global magmaFloatComplex *V, int V_offset, int ldv,
                    __global magmaFloatComplex *c, int c_offset,
                    __global magmaFloatComplex *dwork, int dwork_offset,
                    __global magmaFloatComplex *tau, int tau_offset)
{
    V += V_offset;
    c += c_offset;
    dwork += dwork_offset;
    tau += tau_offset;
    
    const int i = get_local_id(0);
    //const magmaFloatComplex *dV = V + (get_group_id(0)) * ldv;
    V += (get_group_id(0)) * ldv;
    
    __local magmaFloatComplex sum[ BLOCK_SIZE ];
    sum[i] = MAGMA_C_ZERO;
    
    magmaFloatComplex lsum;
    
    if (i == 0)
        c[0] = MAGMA_C_ONE;
    
    /*  lsum := v' * C  */
    lsum = MAGMA_C_ZERO;
    for( int j = i; j < m; j += BLOCK_SIZE )
        lsum += MAGMA_C_MUL( MAGMA_C_CNJG( V[j] ), c[j] );
    
    sum[i] = lsum;
    magma_csum_reduce( BLOCK_SIZE, i, sum );
    
    barrier(CLK_LOCAL_MEM_FENCE);
    if (i == 0)
        dwork [get_group_id(0)] = -tau[0]*sum[0];
}
