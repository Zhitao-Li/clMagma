/*
    -- clMAGMA (version 1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @generated from clmagmablas/zlarfbx.cl, normal z -> d, Tue Jun 18 16:14:14 2019

*/
#include "kernels_header.h"
#include "reduce.h"

//#define BLOCK_SIZE 768
#define BLOCK_SIZE 256


//==============================================================================
__kernel void
magma_dgemv_kernel1(int m, __global double *V, int V_offset, int ldv,
                    __global double *c, int c_offset,
                    __global double *dwork, int dwork_offset)
{
    V += V_offset;
    c += c_offset;
    dwork += dwork_offset;
    
    const int i = get_local_id(0);
    //const double *dV = V + (get_group_id(0)) * ldv;
    V += (get_group_id(0)) * ldv;
    
    __local double sum[ BLOCK_SIZE ];
    double lsum;
    
    /*  lsum := v' * C  */
    lsum = MAGMA_D_ZERO;
    for( int j = i; j < m; j += BLOCK_SIZE )
        lsum += MAGMA_D_MUL( MAGMA_D_CNJG( V[j] ), c[j] );
    
    sum[i] = lsum;
    magma_dsum_reduce( BLOCK_SIZE, i, sum );
    
    barrier(CLK_LOCAL_MEM_FENCE);
    if (i == 0)
        dwork [get_group_id(0)] = sum[0];
}


//==============================================================================
__kernel void
magma_dgemv_kernel2(int m, int n, __global double *V, int V_offset, int ldv,
                    __global double *x, int x_offset,
                    __global double *c, int c_offset)
{
    V += V_offset;
    x += x_offset;
    c += c_offset;
    
    const int i = get_local_id(0);
    const int j = i + BLOCK_SIZE * get_group_id(0);
    double lsum;
    
    V += j;
    
    lsum = MAGMA_D_ZERO;
    if (j < m) {
        for(int k=0; k<n; k++)
            lsum += MAGMA_D_MUL( V[k*ldv], x[k]);
        
        c[j] -= lsum;
    }
}


//==============================================================================
__kernel void
magma_dgemv_kernel3(int m, __global double *V, int V_offset, int ldv,
                    __global double *c, int c_offset,
                    __global double *dwork, int dwork_offset,
                    __global double *tau, int tau_offset)
{
    V += V_offset;
    c += c_offset;
    dwork += dwork_offset;
    tau += tau_offset;
    
    const int i = get_local_id(0);
    //const double *dV = V + (get_group_id(0)) * ldv;
    V += (get_group_id(0)) * ldv;
    
    __local double sum[ BLOCK_SIZE ];
    sum[i] = MAGMA_D_ZERO;
    
    double lsum;
    
    if (i == 0)
        c[0] = MAGMA_D_ONE;
    
    /*  lsum := v' * C  */
    lsum = MAGMA_D_ZERO;
    for( int j = i; j < m; j += BLOCK_SIZE )
        lsum += MAGMA_D_MUL( MAGMA_D_CNJG( V[j] ), c[j] );
    
    sum[i] = lsum;
    magma_dsum_reduce( BLOCK_SIZE, i, sum );
    
    barrier(CLK_LOCAL_MEM_FENCE);
    if (i == 0)
        dwork [get_group_id(0)] = -tau[0]*sum[0];
}
