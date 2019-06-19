/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @generated from clmagmablas/zlarfg.cpp, normal z -> d, Tue Jun 18 16:14:18 2019

       auto-converted from dlarfg.cu
       
       @author Mark Gates
*/
#include "clmagma_runtime.h"
#include "common_magma.h"
#include "dlarfg.h"


/**
    Purpose
    -------
    DLARFG generates a real elementary reflector (Householder matrix)
    H of order n, such that

         H * ( alpha ) = ( beta ),   H**H * H = I.
             (   x   )   (   0  )

    where alpha and beta are scalars, with beta real and beta = ±norm([alpha, x]),
    and x is an (n-1)-element real vector. H is represented in the form

         H = I - tau * ( 1 ) * ( 1 v**H ),
                       ( v )

    where tau is a real scalar and v is a real (n-1)-element vector.
    Note that H is not symmetric.

    If the elements of x are all zero and dalpha is real, then tau = 0
    and H is taken to be the unit matrix.

    Otherwise  1 <= real(tau) <= 2  and  abs(tau-1) <= 1.

    Arguments
    ---------
    @param[in]
    n       INTEGER
            The order of the elementary reflector.

    @param[in,out]
    dalpha  DOUBLE_PRECISION* on the GPU.
            On entry, pointer to the value alpha, i.e., the first entry of the vector.
            On exit, it is overwritten with the value beta.

    @param[in,out]
    dx      DOUBLE_PRECISION array, dimension (1+(N-2)*abs(INCX)), on the GPU
            On entry, the (n-1)-element vector x.
            On exit, it is overwritten with the vector v.

    @param[in]
    incx    INTEGER
            The increment between elements of X. INCX > 0.

    @param[out]
    dtau    DOUBLE_PRECISION* on the GPU.
            Pointer to the value tau.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_daux1
    ********************************************************************/
extern "C"
void magmablas_dlarfg(
    magma_int_t n,
    magmaDouble_ptr dalpha, size_t dalpha_offset,
    magmaDouble_ptr dx, size_t dx_offset, magma_int_t incx,
    magmaDouble_ptr dtau, size_t dtau_offset,
    magma_queue_t queue )
{
    cl_kernel kernel;
    cl_int err;
    int arg;

    const int ndim = 1;
    size_t threads[ndim];
    threads[0] = NB;
    size_t blocks[ndim];
    blocks[0] = 1;
    blocks[0] *= threads[0];
    kernel = g_runtime.get_kernel( "dlarfg_kernel" );
    if ( kernel != NULL ) {
        err = 0;
        arg = 0;
        err |= clSetKernelArg( kernel, arg++, sizeof(n            ), &n             );
        err |= clSetKernelArg( kernel, arg++, sizeof(dalpha       ), &dalpha        );
        err |= clSetKernelArg( kernel, arg++, sizeof(dalpha_offset), &dalpha_offset );
        err |= clSetKernelArg( kernel, arg++, sizeof(dx           ), &dx            );
        err |= clSetKernelArg( kernel, arg++, sizeof(dx_offset    ), &dx_offset     );
        err |= clSetKernelArg( kernel, arg++, sizeof(incx         ), &incx          );
        err |= clSetKernelArg( kernel, arg++, sizeof(dtau         ), &dtau          );
        err |= clSetKernelArg( kernel, arg++, sizeof(dtau_offset  ), &dtau_offset   );
        check_error( err );

        err = clEnqueueNDRangeKernel( queue, kernel, ndim, NULL, blocks, threads, 0, NULL, NULL );
        check_error( err );
    }
}
