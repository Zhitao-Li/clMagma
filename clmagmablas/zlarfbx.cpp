/*
    -- clMAGMA (version 1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c

*/

#include <stdio.h>

#include "magmablas.h"
#include "clmagma_runtime.h"
#include "common_magma.h"

//#define BLOCK_SIZE 768
#define BLOCK_SIZE 256

/*
    Apply a complex block reflector H to a complex vector C from the left
    (i.e., C = H C). H is represented in the form
          H = I - V T V'
    where T is the complex k-by-k upper triangular matrix in the 
    representation of the block reflector, and V is a complex block of
    k elementary reflectors. 
*/
extern "C" magma_int_t
magma_zlarfbx_gpu(int m, int k, magmaDoubleComplex_ptr V, size_t V_offset, int ldv,
                  magmaDoubleComplex_ptr T, size_t T_offset, int ldt, 
                  magmaDoubleComplex_ptr c, size_t c_offset, 
                  magmaDoubleComplex_ptr dwork, size_t dwork_offset, 
                  magma_queue_t queue)
{
    cl_int ciErrNum;                // Error code var
    cl_kernel kernel=NULL;

    kernel = g_runtime.get_kernel( "magma_zgemv_kernel1" );   // in zlarfbx.cl
    if (!kernel)
    {
        printf ("Error: cannot locate kernel in line %d, file %s\n", __LINE__, __FILE__);
        return MAGMA_ERR_UNKNOWN;
    }
    
    int nn = 0;
    ciErrNum  = clSetKernelArg( kernel, nn++, sizeof(int), (void*)&m );
    ciErrNum |= clSetKernelArg( kernel, nn++, sizeof(magmaDoubleComplex_ptr), (void*)&V );
    ciErrNum |= clSetKernelArg( kernel, nn++, sizeof(int), (void*)&V_offset );
    ciErrNum |= clSetKernelArg( kernel, nn++, sizeof(int), (void*)&ldv   );
    ciErrNum |= clSetKernelArg( kernel, nn++, sizeof(magmaDoubleComplex_ptr), (void*)&c );
    ciErrNum |= clSetKernelArg( kernel, nn++, sizeof(int), (void*)&c_offset    );
    ciErrNum |= clSetKernelArg( kernel, nn++, sizeof(magmaDoubleComplex_ptr), (void*)&dwork );
    ciErrNum |= clSetKernelArg( kernel, nn++, sizeof(int), (void*)&dwork_offset    );
    if (ciErrNum != CL_SUCCESS)
    {
        printf("Error: clSetKernelArg at %d in file %s!\n", __LINE__, __FILE__);
        return MAGMA_ERR_UNKNOWN;
    }
    
    size_t GlobalWorkSize[1]={0}, LocalWorkSize[1]={0};
    
    LocalWorkSize[0] = BLOCK_SIZE;
    
    GlobalWorkSize[0] = k*LocalWorkSize[0];
    
    /* dwork = V' c                   */
    //magma_zgemv_kernel1<<< k, BLOCK_SIZE, 0, magma_stream >>>(m, V, ldv, c, dwork); 
    
    // launch kernel magma_zgemv_kernel1
    ciErrNum = clEnqueueNDRangeKernel(
        queue, kernel, 1, NULL, GlobalWorkSize, LocalWorkSize, 0, NULL, NULL);
    if (ciErrNum != CL_SUCCESS)
    {
        printf("Error: clEnqueueNDRangeKernel at %d in file %s \"%s\"\n",
            __LINE__, __FILE__, magma_strerror(ciErrNum));
        return MAGMA_ERR_UNKNOWN;
    }

    //clFlush(queue);
    clFinish(queue);
    
    kernel = g_runtime.get_kernel( "magma_ztrmv_tkernel" );   // in zlarfx.cl
    if (!kernel)
    {
        printf ("Error: cannot locate kernel in line %d, file %s\n", __LINE__, __FILE__);
        return MAGMA_ERR_UNKNOWN;
    }
    
    nn = 0;
    ciErrNum  = clSetKernelArg( kernel, nn++, sizeof(magmaDoubleComplex_ptr), (void*)&T );
    ciErrNum |= clSetKernelArg( kernel, nn++, sizeof(int), (void*)&T_offset );
    ciErrNum |= clSetKernelArg( kernel, nn++, sizeof(int), (void*)&ldt );
    ciErrNum |= clSetKernelArg( kernel, nn++, sizeof(magmaDoubleComplex_ptr), (void*)&dwork   );
    ciErrNum |= clSetKernelArg( kernel, nn++, sizeof(int), (void*)&dwork_offset );
    magmaDoubleComplex_ptr dwork2 = dwork;
    size_t dwork2_offset = dwork_offset + k;
    ciErrNum |= clSetKernelArg( kernel, nn++, sizeof(magmaDoubleComplex_ptr), (void*)&dwork2 );
    ciErrNum |= clSetKernelArg( kernel, nn++, sizeof(int), (void*)&dwork2_offset    );
    if (ciErrNum != CL_SUCCESS)
    {
        printf("Error: clSetKernelArg at %d in file %s!\n", __LINE__, __FILE__);
        return MAGMA_ERR_UNKNOWN;
    }
    
    LocalWorkSize[0] = k;
    GlobalWorkSize[0] = k*LocalWorkSize[0];
    
    /* dwork = T' dwork               */
    //magma_ztrmv_tkernel<<< k, k, 0, magma_stream >>>( T, ldt, dwork, dwork+k);
    
    // launch kernel magma_ztrmv_tkernel
    ciErrNum = clEnqueueNDRangeKernel(
        queue, kernel, 1, NULL, GlobalWorkSize, LocalWorkSize, 0, NULL, NULL);
    if (ciErrNum != CL_SUCCESS)
    {
        printf("Error: clEnqueueNDRangeKernel at %d in file %s \"%s\"\n",
            __LINE__, __FILE__, magma_strerror(ciErrNum));
        return MAGMA_ERR_UNKNOWN;
    }

    //clFlush(queue);
    clFinish(queue);
    
    LocalWorkSize[0] = BLOCK_SIZE;
    GlobalWorkSize[0] = magma_ceildiv( m, BLOCK_SIZE )*LocalWorkSize[0];

    kernel = g_runtime.get_kernel( "magma_zgemv_kernel2" );   // in zlarfbx.cl
    if (!kernel)
    {
        printf ("Error: cannot locate kernel in line %d, file %s\n", __LINE__, __FILE__);
        return MAGMA_ERR_UNKNOWN;
    }
    
    nn = 0;
    ciErrNum  = clSetKernelArg( kernel, nn++, sizeof(int), (void*)&m );
    ciErrNum |= clSetKernelArg( kernel, nn++, sizeof(int), (void*)&k );
    ciErrNum |= clSetKernelArg( kernel, nn++, sizeof(magmaDoubleComplex_ptr), (void*)&V );
    ciErrNum |= clSetKernelArg( kernel, nn++, sizeof(int), (void*)&V_offset   );
    ciErrNum |= clSetKernelArg( kernel, nn++, sizeof(int), (void*)&ldv );
    dwork2 = dwork;
    dwork2_offset = dwork_offset + k;
    ciErrNum |= clSetKernelArg( kernel, nn++, sizeof(magmaDoubleComplex_ptr), (void*)&dwork2 );
    ciErrNum |= clSetKernelArg( kernel, nn++, sizeof(int), (void*)&dwork2_offset    );
    ciErrNum |= clSetKernelArg( kernel, nn++, sizeof(magmaDoubleComplex_ptr), (void*)&c );
    ciErrNum |= clSetKernelArg( kernel, nn++, sizeof(int), (void*)&c_offset    );
    if (ciErrNum != CL_SUCCESS)
    {
        printf("Error: clSetKernelArg at %d in file %s!\n", __LINE__, __FILE__);
        return MAGMA_ERR_UNKNOWN;
    }
    
    // launch kernel magma_zgemv_kernel2
    /* c = c - V dwork                */
    //magma_zgemv_kernel2<<< blocks3, threads3, 0, magma_stream >>>( m, k, V, ldv, dwork+k, c);
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
