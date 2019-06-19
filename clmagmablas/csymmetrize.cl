/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @generated from clmagmablas/zsymmetrize.cl, normal z -> c, Tue Jun 18 16:14:14 2019

       auto-converted from csymmetrize.cu
       @author Mark Gates
*/
#include "kernels_header.h"
#include "csymmetrize.h"

/*
    Matrix is m x m, and is divided into block rows, each NB x m.
    Each block has NB threads.
    Each thread copies one row, iterating across all columns below diagonal.
    The bottom block of rows may be partially outside the matrix;
    if so, rows outside the matrix (i >= m) are disabled.
*/
__kernel void
csymmetrize_lower( magma_int_t m, __global magmaFloatComplex *dA, unsigned long dA_offset, magma_int_t ldda )
{
    dA += dA_offset;

    // dA iterates across row i and dAT iterates down column i.
    int i = get_group_id(0)*NB + get_local_id(0);
    __global magmaFloatComplex *dAT = dA;
    if ( i < m ) {
        dA  += i;
        dAT += i*ldda;
        __global magmaFloatComplex *dAend = dA + i*ldda;
        while( dA < dAend ) {
            *dAT = MAGMA_C_CNJG(*dA);  // upper := lower
            dA  += ldda;
            dAT += 1;
        }
    }
}


// only difference with _lower version is direction dA=dAT instead of dAT=dA.
__kernel void
csymmetrize_upper( magma_int_t m, __global magmaFloatComplex *dA, unsigned long dA_offset, magma_int_t ldda )
{
    dA += dA_offset;

    // dA iterates across row i and dAT iterates down column i.
    int i = get_group_id(0)*NB + get_local_id(0);
    __global magmaFloatComplex *dAT = dA;
    if ( i < m ) {
        dA  += i;
        dAT += i*ldda;
        __global magmaFloatComplex *dAend = dA + i*ldda;
        while( dA < dAend ) {
            *dA = MAGMA_C_CNJG(*dAT);  // lower := upper
            dA  += ldda;
            dAT += 1;
        }
    }
}
