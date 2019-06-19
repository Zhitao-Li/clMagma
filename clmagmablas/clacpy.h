#ifndef MAGMA_CLACPY_H
#define MAGMA_CLACPY_H

/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Mark Gates
       @author Azzam Haidar
       
       @generated from clmagmablas/zlacpy.h, normal z -> c, Tue Jun 18 16:14:13 2019

       auto-converted from clacpy.cu

*/

// BLK_X and BLK_Y need to be equal for claset_q to deal with diag & offdiag
// when looping over super blocks.
// Formerly, BLK_X and BLK_Y could be different.
#define BLK_X 64
#define BLK_Y BLK_X

#endif // MAGMA_CLACPY_H
