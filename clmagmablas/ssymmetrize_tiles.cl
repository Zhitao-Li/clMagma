/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @generated from clmagmablas/zsymmetrize_tiles.cl, normal z -> s, Tue Jun 18 16:14:14 2019

       auto-converted from ssymmetrize_tiles.cu
       @author Mark Gates
*/
#include "kernels_header.h"
#include "ssymmetrize_tiles.h"

/*
    Symmetrizes ntile tiles at a time, e.g., all diagonal tiles of a matrix.
    Grid is ntile x ceil(m/NB).
    Each tile is m x m, and is divided into block rows, each NB x m.
    Each block has NB threads.
    Each thread copies one row, iterating across all columns below diagonal.
    The bottom block of rows may be partially outside the matrix;
    if so, rows outside the matrix (i >= m) are disabled.
*/
__kernel void
ssymmetrize_tiles_lower( magma_int_t m, __global float *dA, unsigned long dA_offset, magma_int_t ldda, magma_int_t mstride, magma_int_t nstride )
{
    dA += dA_offset;

    // shift dA to tile's top-left corner
    dA += get_group_id(1)*(mstride + nstride*ldda);
    
    // dA iterates across row i and dAT iterates down column i.
    int i = get_group_id(0)*NB + get_local_id(0);
    __global float *dAT = dA;
    if ( i < m ) {
        dA  += i;
        dAT += i*ldda;
        __global float *dAend = dA + i*ldda;
        while( dA < dAend ) {
            *dAT = MAGMA_S_CNJG(*dA);  // upper := lower
            dA  += ldda;
            dAT += 1;
        }
    }
}


// only difference with _lower version is direction dA=dAT instead of dAT=dA.
__kernel void
ssymmetrize_tiles_upper( magma_int_t m, __global float *dA, unsigned long dA_offset, magma_int_t ldda, magma_int_t mstride, magma_int_t nstride )
{
    dA += dA_offset;

    // shift dA to tile's top-left corner
    dA += get_group_id(1)*(mstride + nstride*ldda);
    
    // dA iterates across row i and dAT iterates down column i.
    int i = get_group_id(0)*NB + get_local_id(0);
    __global float *dAT = dA;
    if ( i < m ) {
        dA  += i;
        dAT += i*ldda;
        __global float *dAend = dA + i*ldda;
        while( dA < dAend ) {
            *dA  = MAGMA_S_CNJG(*dAT);  // lower := upper
            dA  += ldda;
            dAT += 1;
        }
    }
}
