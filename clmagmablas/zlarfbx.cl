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

//#define BLOCK_SIZE 768
#define BLOCK_SIZE 256


//==============================================================================
__kernel void
magma_zgemv_kernel1(int m, __global magmaDoubleComplex *V, int V_offset, int ldv,
                    __global magmaDoubleComplex *c, int c_offset,
                    __global magmaDoubleComplex *dwork, int dwork_offset)
{
    V += V_offset;
    c += c_offset;
    dwork += dwork_offset;
    
    const int i = get_local_id(0);
    //const magmaDoubleComplex *dV = V + (get_group_id(0)) * ldv;
    V += (get_group_id(0)) * ldv;
    
    __local magmaDoubleComplex sum[ BLOCK_SIZE ];
    magmaDoubleComplex lsum;
    
    /*  lsum := v' * C  */
    lsum = MAGMA_Z_ZERO;
    for( int j = i; j < m; j += BLOCK_SIZE )
        lsum += MAGMA_Z_MUL( MAGMA_Z_CNJG( V[j] ), c[j] );
    
    sum[i] = lsum;
    magma_zsum_reduce( BLOCK_SIZE, i, sum );
    
    barrier(CLK_LOCAL_MEM_FENCE);
    if (i == 0)
        dwork [get_group_id(0)] = sum[0];
}


//==============================================================================
__kernel void
magma_zgemv_kernel2(int m, int n, __global magmaDoubleComplex *V, int V_offset, int ldv,
                    __global magmaDoubleComplex *x, int x_offset,
                    __global magmaDoubleComplex *c, int c_offset)
{
    V += V_offset;
    x += x_offset;
    c += c_offset;
    
    const int i = get_local_id(0);
    const int j = i + BLOCK_SIZE * get_group_id(0);
    magmaDoubleComplex lsum;
    
    V += j;
    
    lsum = MAGMA_Z_ZERO;
    if (j < m) {
        for(int k=0; k<n; k++)
            lsum += MAGMA_Z_MUL( V[k*ldv], x[k]);
        
        c[j] -= lsum;
    }
}


//==============================================================================
__kernel void
magma_zgemv_kernel3(int m, __global magmaDoubleComplex *V, int V_offset, int ldv,
                    __global magmaDoubleComplex *c, int c_offset,
                    __global magmaDoubleComplex *dwork, int dwork_offset,
                    __global magmaDoubleComplex *tau, int tau_offset)
{
    V += V_offset;
    c += c_offset;
    dwork += dwork_offset;
    tau += tau_offset;
    
    const int i = get_local_id(0);
    //const magmaDoubleComplex *dV = V + (get_group_id(0)) * ldv;
    V += (get_group_id(0)) * ldv;
    
    __local magmaDoubleComplex sum[ BLOCK_SIZE ];
    sum[i] = MAGMA_Z_ZERO;
    
    magmaDoubleComplex lsum;
    
    if (i == 0)
        c[0] = MAGMA_Z_ONE;
    
    /*  lsum := v' * C  */
    lsum = MAGMA_Z_ZERO;
    for( int j = i; j < m; j += BLOCK_SIZE )
        lsum += MAGMA_Z_MUL( MAGMA_Z_CNJG( V[j] ), c[j] );
    
    sum[i] = lsum;
    magma_zsum_reduce( BLOCK_SIZE, i, sum );
    
    barrier(CLK_LOCAL_MEM_FENCE);
    if (i == 0)
        dwork [get_group_id(0)] = -tau[0]*sum[0];
}
