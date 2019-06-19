/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @generated from clmagmablas/zsymmetrize_tiles.cpp, normal z -> c, Tue Jun 18 16:14:18 2019

       auto-converted from csymmetrize_tiles.cu
       @author Mark Gates
*/
#include "clmagma_runtime.h"
#include "common_magma.h"
#include "csymmetrize_tiles.h"


/**
    Purpose
    -------
    
    CSYMMETRIZE_TILES copies lower triangle to upper triangle, or vice-versa,
    to make some blocks of dA into general representations of a symmetric block.
    This processes NTILE blocks, typically the diagonal blocks.
    Each block is offset by mstride rows and nstride columns from the previous block.
    
    Arguments
    ---------
    
    @param[in]
    uplo    magma_uplo_t
            Specifies the part of the matrix dA that is valid on input.
      -     = MagmaUpper:      Upper triangular part
      -     = MagmaLower:      Lower triangular part
    
    @param[in]
    m       INTEGER
            The number of rows & columns of each square block of dA.  M >= 0.
    
    @param[in,out]
    dA      COMPLEX array, dimension (LDDA,N)
            The matrix dA. N = m + nstride*(ntile-1).
    
    @param[in]
    ldda    INTEGER
            The leading dimension of the array dA.  LDDA >= max(1, m + mstride*(ntile-1)).
    
    @param[in]
    ntile   INTEGER
            Number of blocks to symmetrize. ntile >= 0.
    
    @param[in]
    mstride INTEGER
            Row offset from start of one block to start of next block. mstride >= 0.
            Either (mstride >= m) or (nstride >= m), to prevent m-by-m tiles
            from overlapping.
    
    @param[in]
    nstride INTEGER
            Column offset from start of one block to start of next block. nstride >= 0.
    
    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_caux2
    ********************************************************************/
extern "C" void
magmablas_csymmetrize_tiles(
    magma_uplo_t uplo, magma_int_t m,
    magmaFloatComplex_ptr dA, size_t dA_offset, magma_int_t ldda,
    magma_int_t ntile, magma_int_t mstride, magma_int_t nstride,
    magma_queue_t queue )
{
    cl_kernel kernel;
    cl_int err;
    int arg;

    magma_int_t info = 0;
    if ( uplo != MagmaLower && uplo != MagmaUpper )
        info = -1;
    else if ( m < 0 )
        info = -2;
    else if ( ldda < max(1,m + mstride*(ntile-1)) )
        info = -5;
    else if ( ntile < 0 )
        info = -6;
    else if ( mstride < 0 )
        info = -7;
    else if ( nstride < 0 )
        info = -8;
    else if ( mstride < m && nstride < m )  // only one must be >= m.
        info = -7;
    
    if ( info != 0 ) {
        magma_xerbla( __func__, -(info) );
        return;
    }
    
    if ( m == 0 || ntile == 0 )
        return;
    
    const int ndim = 2;
    size_t threads[ndim];
    threads[0] = NB;
    threads[1] = 1;
    size_t grid[ndim];
    grid[0] = magma_ceildiv( m, NB );
    grid[1] = ntile;
    grid[0] *= threads[0];
    grid[1] *= threads[1];
    
    //printf( "m %d, grid %d x %d, threads %d\n", m, grid.x, grid.y, threads.x );
    if ( uplo == MagmaUpper ) {
        kernel = g_runtime.get_kernel( "csymmetrize_tiles_upper" );
        if ( kernel != NULL ) {
            err = 0;
            arg = 0;
            err |= clSetKernelArg( kernel, arg++, sizeof(m        ), &m         );
            err |= clSetKernelArg( kernel, arg++, sizeof(dA       ), &dA        );
            err |= clSetKernelArg( kernel, arg++, sizeof(dA_offset), &dA_offset );
            err |= clSetKernelArg( kernel, arg++, sizeof(ldda     ), &ldda      );
            err |= clSetKernelArg( kernel, arg++, sizeof(mstride  ), &mstride   );
            err |= clSetKernelArg( kernel, arg++, sizeof(nstride  ), &nstride   );
            check_error( err );

            err = clEnqueueNDRangeKernel( queue, kernel, ndim, NULL, grid, threads, 0, NULL, NULL );
            check_error( err );
        }
    }
    else {
        kernel = g_runtime.get_kernel( "csymmetrize_tiles_lower" );
        if ( kernel != NULL ) {
            err = 0;
            arg = 0;
            err |= clSetKernelArg( kernel, arg++, sizeof(m        ), &m         );
            err |= clSetKernelArg( kernel, arg++, sizeof(dA       ), &dA        );
            err |= clSetKernelArg( kernel, arg++, sizeof(dA_offset), &dA_offset );
            err |= clSetKernelArg( kernel, arg++, sizeof(ldda     ), &ldda      );
            err |= clSetKernelArg( kernel, arg++, sizeof(mstride  ), &mstride   );
            err |= clSetKernelArg( kernel, arg++, sizeof(nstride  ), &nstride   );
            check_error( err );

            err = clEnqueueNDRangeKernel( queue, kernel, ndim, NULL, grid, threads, 0, NULL, NULL );
            check_error( err );
        }
    }
}
