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

#define BLOCK_SIZEx  32
#define BLOCK_SIZEy  8


__kernel void magma_ztrmv_tkernel(__global magmaDoubleComplex *T, int T_offset, int ldt, __global magmaDoubleComplex *t, int t_offset, 
                                  __global magmaDoubleComplex *y, int y_offset)
{
    T += T_offset;
    t += t_offset;
    y += y_offset;

    const int i = get_local_id(0);
    T += get_group_id(0)*ldt;
    
    __local magmaDoubleComplex sum[ 128 ];
    
    sum[i] = MAGMA_Z_CNJG(T[i])*t[i];
    magma_zsum_reduce( get_local_size(0), i, sum );
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    if (i==0)
        y[get_group_id(0)] = sum[0];
}


__kernel 
void magma_ztrmv_kernel2(__global magmaDoubleComplex *T, int T_offset, int ldt, __global magmaDoubleComplex *t, int t_offset, 
                         __global magmaDoubleComplex *y, int y_offset, __global magmaDoubleComplex *tau, int tau_offset)
{
    T += T_offset;
    t += t_offset;
    y += y_offset;
    tau += tau_offset;

    const int i = get_local_id(0);
    T += get_group_id(0);

    __local magmaDoubleComplex sum[ 128 ];

    sum[i] = T[i*ldt]*t[i];
    magma_zsum_reduce( get_local_size(0), i, sum );

    barrier(CLK_LOCAL_MEM_FENCE);

    if (i==0){
        y[get_group_id(0)] = sum[0];
        if (get_group_id(0)==0)
            y[get_num_groups(0)] = tau[0];
    }
}
