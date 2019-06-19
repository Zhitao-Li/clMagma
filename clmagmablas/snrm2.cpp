/*
    -- clMAGMA (version 1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @generated from clmagmablas/dznrm2.cpp, normal z -> s, Tue Jun 18 16:14:18 2019

*/
#include <stdio.h>

#include "clmagma_runtime.h"
#include "common_magma.h"

//#define BLOCK_SIZE  512
#define BLOCK_SIZE  256
#define BLOCK_SIZEx  32
//#define BLOCK_SIZEy  16
#define BLOCK_SIZEy  8

#define PRECISION_s

/*
    Adjust the norm of c to give the norm of c[k+1:], assumin that
    c was changed with orthogonal transformations.
*/
extern "C" magma_int_t
magmablas_snrm2_adjust(int k, magmaFloat_ptr xnorm, size_t xnorm_offset, magmaFloat_ptr c, size_t c_offset, magma_queue_t queue)
{
    size_t GlobalWorkSize[1]={0}, LocalWorkSize[1]={0};
    
    LocalWorkSize[0] = k;
    GlobalWorkSize[0] = 1*LocalWorkSize[0];
    
    cl_int ciErrNum;                // Error code var
    cl_kernel kernel=NULL;
    
    kernel = g_runtime.get_kernel( "magmablas_snrm2_adjust_kernel" );
    if (!kernel)
    {
        printf ("Error: cannot locate kernel in line %d, file %s\n", __LINE__, __FILE__);
        return MAGMA_ERR_UNKNOWN;
    }
    
    int nn = 0;
    ciErrNum  = clSetKernelArg( kernel, nn++, sizeof(magmaFloat_ptr), (void*)&xnorm   );
    ciErrNum |= clSetKernelArg( kernel, nn++, sizeof(int), (void*)&xnorm_offset );
    ciErrNum |= clSetKernelArg( kernel, nn++, sizeof(magmaFloat_ptr), (void*)&c     );
    ciErrNum |= clSetKernelArg( kernel, nn++, sizeof(int), (void*)&c_offset   );
    //magma_snrm2_adjust_kernel<<< 1, k, 0, magma_stream >>> (xnorm, c);
    // launch kernel
    ciErrNum = clEnqueueNDRangeKernel(
        queue, kernel, 1, NULL, GlobalWorkSize, LocalWorkSize, 0, NULL, NULL);
    if (ciErrNum != CL_SUCCESS)
    {
        printf("Error: clEnqueueNDRangeKernel at %d in file %s \"%s\"\n",
            __LINE__, __FILE__, magma_strerror(ciErrNum));
        printf("block: %d,    group: %d\n", (int) LocalWorkSize[0], (int) GlobalWorkSize[0]);
        return MAGMA_ERR_UNKNOWN;
    }

    clFlush(queue);
    return MAGMA_SUCCESS;
}
//==============================================================================

/*
   Compute the snrm2 of da, da+ldda, ..., da +(num-1)*ldda where the vectors are
   of size m. The resulting norms are written in the dxnorm array. 
   The computation can be done using num blocks (default) or on one SM (commented).
*/
extern "C" magma_int_t
magmablas_snrm2(int m, int num, magmaFloat_ptr da, size_t da_offset, magma_int_t ldda, 
                 magmaFloat_ptr dxnorm, size_t dxnorm_offset, magma_queue_t queue) 
{
    size_t GlobalWorkSize[1]={0}, LocalWorkSize[1]={0};
   
    LocalWorkSize[0] = BLOCK_SIZE;
    GlobalWorkSize[0] = num * LocalWorkSize[0];
    
    cl_int ciErrNum;                // Error code var
    cl_kernel kernel=NULL;
    
    kernel = g_runtime.get_kernel( "magmablas_snrm2_kernel" );
    if (!kernel)
    {
        printf ("Error: cannot locate kernel in line %d, file %s\n", __LINE__, __FILE__);
        return MAGMA_ERR_UNKNOWN;
    }
    
    int nn = 0;
    ciErrNum  = clSetKernelArg( kernel, nn++, sizeof(int), (void*)&m   );
    ciErrNum |= clSetKernelArg( kernel, nn++, sizeof(magmaFloat_ptr), (void*)&da );
    ciErrNum |= clSetKernelArg( kernel, nn++, sizeof(int), (void*)&da_offset     );
    ciErrNum |= clSetKernelArg( kernel, nn++, sizeof(int), (void*)&ldda   );
    ciErrNum |= clSetKernelArg( kernel, nn++, sizeof(magmaFloat_ptr), (void*)&dxnorm );
    ciErrNum |= clSetKernelArg( kernel, nn++, sizeof(int), (void*)&dxnorm_offset     );
    if (ciErrNum != CL_SUCCESS)
    {
        printf("Error: clSetKernelArg at %d in file %s!\n", __LINE__, __FILE__);
        return MAGMA_ERR_UNKNOWN;
    }
    
    // launch kernel
    ciErrNum = clEnqueueNDRangeKernel(
        queue, kernel, 1, NULL, GlobalWorkSize, LocalWorkSize, 0, NULL, NULL);
    if (ciErrNum != CL_SUCCESS)
    {
        printf("Error: clEnqueueNDRangeKernel at %d in file %s \"%s\"\n",
            __LINE__, __FILE__, magma_strerror(ciErrNum));
        return MAGMA_ERR_UNKNOWN;
    }
    
    clFlush(queue);
    return MAGMA_SUCCESS;
}