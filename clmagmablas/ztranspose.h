#ifndef MAGMA_ZTRANSPOSE_H
#define MAGMA_ZTRANSPOSE_H

/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c

       auto-converted from ztranspose.cu

       @author Stan Tomov
       @author Mark Gates
*/

#define PRECISION_z

#if defined(PRECISION_z)
    #define NX 16
#else
    #define NX 32
#endif

#define NB 32
#define NY 8

#endif // MAGMA_ZTRANSPOSE_H
