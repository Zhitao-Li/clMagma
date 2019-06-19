#ifndef MAGMA_DTRANSPOSE_H
#define MAGMA_DTRANSPOSE_H

/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @generated from clmagmablas/ztranspose.h, normal z -> d, Tue Jun 18 16:14:14 2019

       auto-converted from dtranspose.cu

       @author Stan Tomov
       @author Mark Gates
*/

#define PRECISION_d

#if defined(PRECISION_z)
    #define NX 16
#else
    #define NX 32
#endif

#define NB 32
#define NY 8

#endif // MAGMA_DTRANSPOSE_H
