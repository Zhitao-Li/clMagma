/*
    -- clMAGMA (version 1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @generated from clmagmablas/zlarfgx-v2.cpp, normal z -> s, Tue Jun 18 16:14:19 2019

*/

#include <stdio.h>
#include "clmagma_runtime.h"
#include "common_magma.h"

//#define BLOCK_SIZE 768
#define BLOCK_SIZE 256

#define PRECISION_s

//==============================================================================

/*
    Generates Householder elementary reflector H = I - tau v v^T to reduce
        H [ dx0 ] = [ beta ]
          [ dx  ]   [ 0    ]
    with beta = ±norm( [dx0, dx] ) = ±dxnorm[0].
    Stores v over dx; first element of v is 1 and is not stored.
    Stores beta over dx0.
    Stores tau.  

    The difference with LAPACK's slarfg is that the norm of dx, and hance beta,
    are computed outside the routine and passed to it in dxnorm (array on the GPU).
*/
extern "C" magma_int_t
magma_slarfgx_gpu(
    int n, magmaFloat_ptr dx0, size_t dx0_offset, magmaFloat_ptr dx, size_t dx_offset,  
    magmaFloat_ptr dtau, size_t dtau_offset, magmaFloat_ptr dxnorm, size_t dxnorm_offset,  
    magmaFloat_ptr dA, size_t dA_offset, int it,
    magma_queue_t queue)
{
    cl_int ciErrNum;                // Error code var
    cl_kernel kernel=NULL;
    
    kernel = g_runtime.get_kernel( "magma_slarfgx_gpu_kernel" );
    if ( ! kernel ) {
        printf ("Error: cannot locate kernel in line %d, file %s\n", __LINE__, __FILE__);
        return MAGMA_ERR_UNKNOWN;
    }
    
    int nn = 0;
    ciErrNum  = clSetKernelArg( kernel, nn++, sizeof(int), (void*)&n   );
    ciErrNum |= clSetKernelArg( kernel, nn++, sizeof(magmaFloat_ptr), (void*)&dx0 );
    ciErrNum |= clSetKernelArg( kernel, nn++, sizeof(int), (void*)&dx0_offset     );
    ciErrNum |= clSetKernelArg( kernel, nn++, sizeof(magmaFloat_ptr), (void*)&dx   );
    ciErrNum |= clSetKernelArg( kernel, nn++, sizeof(int), (void*)&dx_offset );
    ciErrNum |= clSetKernelArg( kernel, nn++, sizeof(magmaFloat_ptr), (void*)&dtau     );
    ciErrNum |= clSetKernelArg( kernel, nn++, sizeof(int), (void*)&dtau_offset );
    ciErrNum |= clSetKernelArg( kernel, nn++, sizeof(magmaFloat_ptr), (void*)&dxnorm     );
    ciErrNum |= clSetKernelArg( kernel, nn++, sizeof(int), (void*)&dxnorm_offset );
    ciErrNum |= clSetKernelArg( kernel, nn++, sizeof(magmaFloat_ptr), (void*)&dA     );
    ciErrNum |= clSetKernelArg( kernel, nn++, sizeof(int), (void*)&dA_offset );
    ciErrNum |= clSetKernelArg( kernel, nn++, sizeof(int), (void*)&it );
    if (ciErrNum != CL_SUCCESS) {
        printf("Error: clSetKernelArg at %d in file %s!\n", __LINE__, __FILE__);
        return MAGMA_ERR_UNKNOWN;
    }
    /*
    dim3 blocks( magma_ceildiv( n, BLOCK_SIZE ));
    dim3 threads( BLOCK_SIZE );
    */

    size_t GlobalWorkSize[1]={0}, LocalWorkSize[1]={0};
    
    LocalWorkSize[0] = BLOCK_SIZE;
    GlobalWorkSize[0] = magma_ceildiv( n, BLOCK_SIZE )*LocalWorkSize[0];
    
    // launch kernel
    ciErrNum = clEnqueueNDRangeKernel(
        queue, kernel, 1, NULL, GlobalWorkSize, LocalWorkSize, 0, NULL, NULL);
    if (ciErrNum != CL_SUCCESS) {
        printf("Error: clEnqueueNDRangeKernel at %d in file %s \"%s\"\n",
            __LINE__, __FILE__, magma_strerror(ciErrNum));
        return MAGMA_ERR_UNKNOWN;
    }

    clFlush(queue);
    return MAGMA_SUCCESS;
}

//==============================================================================

/*
    Generates Householder elementary reflector H = I - tau v v^T to reduce
        H [ dx0 ] = [ beta ]
          [ dx  ]   [ 0    ]
    with beta = ±norm( [dx0, dx] ) = ±dxnorm[0].
    Stores v over dx; first element of v is 1 and is not stored.
    Stores beta over dx0.
    Stores tau.

    The difference with LAPACK's slarfg is that the norm of dx, and hance beta,
    are computed outside the routine and passed to it in dxnorm (array on the GPU).
*/
extern "C" magma_int_t
magma_slarfgtx_gpu(
    int n, magmaFloat_ptr dx0, size_t dx0_offset, magmaFloat_ptr dx, size_t dx_offset,
    magmaFloat_ptr dtau, size_t dtau_offset, magmaFloat_ptr dxnorm, size_t dxnorm_offset, 
    magmaFloat_ptr dA, size_t dA_offset, int i, 
    magmaFloat_ptr V, size_t V_offset, int ldv, magmaFloat_ptr T, size_t T_offset, int ldt, 
    magmaFloat_ptr work, size_t work_offset, 
    magma_queue_t queue)
{
    /*  Generate the elementary reflector H(i)  */
    magma_slarfgx_gpu(n, dx0, dx0_offset, dx, dx_offset, dtau, dtau_offset, dxnorm, dxnorm_offset, dA, dA_offset, i, queue);

    if (i == 0) {
        float tt = MAGMA_S_ONE;
        magmablas_slacpy(MagmaFull, 1, 1, dtau, dtau_offset, 1, T, T_offset+i+i*ldt, 1, queue);
        magma_ssetmatrix(1, 1, &tt, 1, dx0, dx0_offset, 1, queue);
    }
    else {
        /* Compute the i-th column of T */      
        cl_int ciErrNum;                // Error code var
        cl_kernel kernel=NULL;
        kernel = g_runtime.get_kernel( "magma_sgemv_kernel3" );     // in slarfbx.cl
      
        if ( ! kernel ) {
            printf ("Error: cannot locate kernel in line %d, file %s\n", __LINE__, __FILE__);
            return MAGMA_ERR_UNKNOWN;
        }
      
        int nn = 0;
        ciErrNum  = clSetKernelArg( kernel, nn++, sizeof(int), (void*)&n   );
        ciErrNum |= clSetKernelArg( kernel, nn++, sizeof(magmaFloat_ptr), (void*)&V );
        ciErrNum |= clSetKernelArg( kernel, nn++, sizeof(int), (void*)&V_offset     );
        ciErrNum |= clSetKernelArg( kernel, nn++, sizeof(int), (void*)&ldv   );
        ciErrNum |= clSetKernelArg( kernel, nn++, sizeof(magmaFloat_ptr), (void*)&dx0 );
        ciErrNum |= clSetKernelArg( kernel, nn++, sizeof(int), (void*)&dx0_offset     );
        ciErrNum |= clSetKernelArg( kernel, nn++, sizeof(magmaFloat_ptr), (void*)&work );
        ciErrNum |= clSetKernelArg( kernel, nn++, sizeof(int), (void*)&work_offset     );
        ciErrNum |= clSetKernelArg( kernel, nn++, sizeof(magmaFloat_ptr), (void*)&dtau );
        ciErrNum |= clSetKernelArg( kernel, nn++, sizeof(int), (void*)&dtau_offset     );
        if (ciErrNum != CL_SUCCESS) {
            printf("Error: clSetKernelArg at %d in file %s!\n", __LINE__, __FILE__);
            return MAGMA_ERR_UNKNOWN;
        }

        size_t GlobalWorkSize[1]={0}, LocalWorkSize[1]={0};
    
        LocalWorkSize[0] = BLOCK_SIZE;
        GlobalWorkSize[0] = i*LocalWorkSize[0];
    
        // launch kernel
        ciErrNum = clEnqueueNDRangeKernel(
            queue, kernel, 1, NULL, GlobalWorkSize, LocalWorkSize, 0, NULL, NULL);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error: clEnqueueNDRangeKernel at %d in file %s \"%s\"\n",
                __LINE__, __FILE__, magma_strerror(ciErrNum));
            return MAGMA_ERR_UNKNOWN;
        }

        //magma_sgemv_kernel3<<< i, BLOCK_SIZE, 0, magma_stream >>>(n, V, ldv, dx0, work, dtau);
        
        clFlush(queue);  
        
        kernel = g_runtime.get_kernel( "magma_strmv_kernel2" );         // in slarfx.cl
      
        if ( ! kernel ) {
            printf ("Error: cannot locate kernel in line %d, file %s\n", __LINE__, __FILE__);
            return MAGMA_ERR_UNKNOWN;
        }

        nn = 0;
        ciErrNum  = clSetKernelArg( kernel, nn++, sizeof(magmaFloat_ptr), (void*)&T   );
        ciErrNum |= clSetKernelArg( kernel, nn++, sizeof(int), (void*)&T_offset     );
        ciErrNum |= clSetKernelArg( kernel, nn++, sizeof(int), (void*)&ldt   );
        ciErrNum |= clSetKernelArg( kernel, nn++, sizeof(magmaFloat_ptr), (void*)&work );
        ciErrNum |= clSetKernelArg( kernel, nn++, sizeof(int), (void*)&work_offset     );
        magmaFloat_ptr T1 = T;
        size_t T1_offset = T_offset + i*ldt;
        ciErrNum |= clSetKernelArg( kernel, nn++, sizeof(magmaFloat_ptr), (void*)&T1 );
        ciErrNum |= clSetKernelArg( kernel, nn++, sizeof(int), (void*)&T1_offset     );
        ciErrNum |= clSetKernelArg( kernel, nn++, sizeof(magmaFloat_ptr), (void*)&dtau );
        ciErrNum |= clSetKernelArg( kernel, nn++, sizeof(int), (void*)&dtau_offset     );
        if (ciErrNum != CL_SUCCESS) {
            printf("Error: clSetKernelArg at %d in file %s!\n", __LINE__, __FILE__);
            return MAGMA_ERR_UNKNOWN;
        }
    
        LocalWorkSize[0] = i;
        GlobalWorkSize[0] = i*LocalWorkSize[0];
    
        // launch kernel
        ciErrNum = clEnqueueNDRangeKernel(
            queue, kernel, 1, NULL, GlobalWorkSize, LocalWorkSize, 0, NULL, NULL);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error: clEnqueueNDRangeKernel at %d in file %s \"%s\"\n",
                __LINE__, __FILE__, magma_strerror(ciErrNum));
            printf("block: %d,    group: %d\n", (int) LocalWorkSize[0], (int) GlobalWorkSize[0]);
            return MAGMA_ERR_UNKNOWN;
        }
      
        //magma_strmv_kernel2<<< i, i, 0, magma_stream          >>>( T, ldt, work, T+i*ldt, dtau);
        clFlush(queue);
    }
    return MAGMA_SUCCESS;
}
//==============================================================================

