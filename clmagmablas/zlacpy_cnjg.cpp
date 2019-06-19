/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c

       auto-converted from zlacpy_cnjg.cu

*/
#include "clmagma_runtime.h"
#include "common_magma.h"
#include "zlacpy_cnjg.h"


extern "C" void 
magmablas_zlacpy_cnjg(
    magma_int_t n,
    magmaDoubleComplex_ptr dA1, size_t dA1_offset, magma_int_t lda1, 
    magmaDoubleComplex_ptr dA2, size_t dA2_offset, magma_int_t lda2,
    magma_queue_t queue )
{
    cl_kernel kernel;
    cl_int err;
    int arg;

    const int ndim = 1;
    size_t threads[ndim];
    threads[0] = BLOCK_SIZE;
    size_t blocks[ndim];
    blocks[0] = magma_ceildiv( n, BLOCK_SIZE );
    blocks[0] *= threads[0];
    kernel = g_runtime.get_kernel( "zlacpy_cnjg_kernel" );
    if ( kernel != NULL ) {
        err = 0;
        arg = 0;
        err |= clSetKernelArg( kernel, arg++, sizeof(n         ), &n          );
        err |= clSetKernelArg( kernel, arg++, sizeof(dA1       ), &dA1        );
        err |= clSetKernelArg( kernel, arg++, sizeof(dA1_offset), &dA1_offset );
        err |= clSetKernelArg( kernel, arg++, sizeof(lda1      ), &lda1       );
        err |= clSetKernelArg( kernel, arg++, sizeof(dA2       ), &dA2        );
        err |= clSetKernelArg( kernel, arg++, sizeof(dA2_offset), &dA2_offset );
        err |= clSetKernelArg( kernel, arg++, sizeof(lda2      ), &lda2       );
        check_error( err );

        err = clEnqueueNDRangeKernel( queue, kernel, ndim, NULL, blocks, threads, 0, NULL, NULL );
        check_error( err );
    }
}
