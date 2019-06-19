#ifndef MAGMA_CTRANSPOSE_H
#define MAGMA_CTRANSPOSE_H

/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @generated from clmagmablas/ztranspose.h, normal z -> c, Tue Jun 18 16:14:14 2019

       auto-converted from ctranspose.cu

       @author Stan Tomov
       @author Mark Gates
*/

#define PRECISION_c

#if defined(PRECISION_z)
    #define NX 16
#else
    #define NX 32
#endif

#define NB 32
#define NY 8

#endif // MAGMA_CTRANSPOSE_H
