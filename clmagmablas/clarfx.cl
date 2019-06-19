/*
    -- clMAGMA (version 1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @generated from clmagmablas/zlarfx.cl, normal z -> c, Tue Jun 18 16:14:14 2019

*/
#include "kernels_header.h"
#include "reduce.h"

//#define BLOCK_SIZE 768
#define BLOCK_SIZE 256

#define BLOCK_SIZEx  32
#define BLOCK_SIZEy  8


__kernel void magma_ctrmv_tkernel(__global magmaFloatComplex *T, int T_offset, int ldt, __global magmaFloatComplex *t, int t_offset, 
                                  __global magmaFloatComplex *y, int y_offset)
{
    T += T_offset;
    t += t_offset;
    y += y_offset;

    const int i = get_local_id(0);
    T += get_group_id(0)*ldt;
    
    __local magmaFloatComplex sum[ 128 ];
    
    sum[i] = MAGMA_C_CNJG(T[i])*t[i];
    magma_csum_reduce( get_local_size(0), i, sum );
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    if (i==0)
        y[get_group_id(0)] = sum[0];
}


__kernel 
void magma_ctrmv_kernel2(__global magmaFloatComplex *T, int T_offset, int ldt, __global magmaFloatComplex *t, int t_offset, 
                         __global magmaFloatComplex *y, int y_offset, __global magmaFloatComplex *tau, int tau_offset)
{
    T += T_offset;
    t += t_offset;
    y += y_offset;
    tau += tau_offset;

    const int i = get_local_id(0);
    T += get_group_id(0);

    __local magmaFloatComplex sum[ 128 ];

    sum[i] = T[i*ldt]*t[i];
    magma_csum_reduce( get_local_size(0), i, sum );

    barrier(CLK_LOCAL_MEM_FENCE);

    if (i==0){
        y[get_group_id(0)] = sum[0];
        if (get_group_id(0)==0)
            y[get_num_groups(0)] = tau[0];
    }
}
