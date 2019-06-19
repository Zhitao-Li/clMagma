#ifndef MAGMA_DLASET_H
#define MAGMA_DLASET_H

/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Mark Gates
       @author Azzam Haidar
       
       @generated from clmagmablas/zlaset.h, normal z -> d, Tue Jun 18 16:14:13 2019

       auto-converted from dlaset.cu

*/

// BLK_X and BLK_Y need to be equal for dlaset_q to deal with diag & offdiag
// when looping over super blocks.
// Formerly, BLK_X and BLK_Y could be different.
#define BLK_X 64
#define BLK_Y BLK_X

#endif // MAGMA_DLASET_H
