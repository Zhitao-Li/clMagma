/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Mark Gates
       @generated from control/zpanel_to_q.cpp, normal z -> c, Tue Jun 18 16:14:15 2019
*/
#include "common_magma.h"

// -------------------------
// Put 0s in the upper triangular part of a panel and 1s on the diagonal.
// Stores previous values in work array, to be restored later with magma_cq_to_panel.
extern "C"
void magma_cpanel_to_q(magma_uplo_t uplo, magma_int_t ib, magmaFloatComplex *A, magma_int_t lda, magmaFloatComplex *work)
{
    magma_int_t i, j, k = 0;
    magmaFloatComplex *col;
    magmaFloatComplex c_zero = MAGMA_C_ZERO;
    magmaFloatComplex c_one  = MAGMA_C_ONE;
    
    if (uplo == MagmaUpper) {
        for(i = 0; i < ib; ++i) {
            col = A + i*lda;
            for(j = 0; j < i; ++j) {
                work[k] = col[j];
                col [j] = c_zero;
                ++k;
            }
            
            work[k] = col[i];
            col [j] = c_one;
            ++k;
        }
    }
    else {
        for(i=0; i<ib; ++i) {
            col = A + i*lda;
            work[k] = col[i];
            col [i] = c_one;
            ++k;
            for(j=i+1; j<ib; ++j) {
                work[k] = col[j];
                col [j] = c_zero;
                ++k;
            }
        }
    }
}


// -------------------------
// Restores a panel, after call to magma_cpanel_to_q.
extern "C"
void magma_cq_to_panel(magma_uplo_t uplo, magma_int_t ib, magmaFloatComplex *A, magma_int_t lda, magmaFloatComplex *work)
{
    magma_int_t i, j, k = 0;
    magmaFloatComplex *col;
    
    if (uplo == MagmaUpper) {
        for(i = 0; i < ib; ++i) {
            col = A + i*lda;
            for(j = 0; j <= i; ++j) {
                col[j] = work[k];
                ++k;
            }
        }
    }
    else {
        for(i = 0; i < ib; ++i) {
            col = A + i*lda;
            for(j = i; j < ib; ++j) {
                col[j] = work[k];
                ++k;
            }
        }
    }
}
