#ifndef MAGMA_STRANSPOSE_H
#define MAGMA_STRANSPOSE_H

/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @generated from clmagmablas/ztranspose.h, normal z -> s, Tue Jun 18 16:14:13 2019

       auto-converted from stranspose.cu

       @author Stan Tomov
       @author Mark Gates
*/

#define PRECISION_s

#if defined(PRECISION_z)
    #define NX 16
#else
    #define NX 32
#endif

#define NB 32
#define NY 8

#endif // MAGMA_STRANSPOSE_H
