/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @generated from clmagmablas/ztranspose.cl, normal z -> s, Tue Jun 18 16:42:02 2019

       auto-converted from stranspose.cu

       @author Stan Tomov
       @author Mark Gates
*/
#include "kernels_header.h"
#include "stranspose.h"


// tile M-by-N matrix with ceil(M/NB) by ceil(N/NB) tiles sized NB-by-NB.
// uses NX-by-NY threads, where NB/NX, NB/NY, NX/NY evenly.
// subtile each NB-by-NB tile with (NB/NX) subtiles sized NX-by-NB
// for each subtile
//     load NX-by-NB subtile transposed from A into sA, as (NB/NY) blocks sized NX-by-NY
//     save NB-by-NX subtile from sA into AT,   as (NB/NX)*(NX/NY) blocks sized NX-by-NY
//     A  += NX
//     AT += NX*ldat
//
// e.g., with NB=32, NX=32, NY=8 ([sdc] precisions)
//     load 32x32 subtile as 4   blocks of 32x8 columns: (A11  A12  A13  A14 )
//     save 32x32 subtile as 1*4 blocks of 32x8 columns: (AT11 AT12 AT13 AT14)
//
// e.g., with NB=32, NX=16, NY=8 (z precision)
//     load 16x32 subtile as 4   blocks of 16x8 columns: (A11  A12  A13  A14)
//     save 32x16 subtile as 2*2 blocks of 16x8 columns: (AT11 AT12)
//                                                       (AT21 AT22)

// prototype to suppress compiler warning
void
stranspose_device(
    magma_int_t m, magma_int_t n,
    __global const float *A, magma_int_t lda,
    __global float *AT,      magma_int_t ldat);

void
stranspose_device(
    magma_int_t m, magma_int_t n,
    __global const float *A, magma_int_t lda,
    __global float *AT,      magma_int_t ldat)
{
    float sA[NB][NX+1];

    int tx  = get_local_id(0);
    int ty  = get_local_id(1);
    int ibx = get_group_id(0)*NB;
    int iby = get_group_id(1)*NB;
    int i, j;
    
    A  += ibx + tx + (iby + ty)*lda;
    AT += iby + tx + (ibx + ty)*ldat;
    
    #pragma unroll
    for( int tile=0; tile < NB/NX; ++tile ) {
        // load NX-by-NB subtile transposed from A into sA
        i = ibx + tx + tile*NX;
        j = iby + ty;
        if (i < m) {
            #pragma unroll
            for( int j2=0; j2 < NB; j2 += NY ) {
                if (j + j2 < n) {
                    sA[ty + j2][tx] = A[j2*lda];
                }
            }
        }
        barrier( CLK_LOCAL_MEM_FENCE );
        
        // save NB-by-NX subtile from sA into AT
        i = iby + tx;
        j = ibx + ty + tile*NX;
        #pragma unroll
        for( int i2=0; i2 < NB; i2 += NX ) {
            if (i + i2 < n) {
                #pragma unroll
                for( int j2=0; j2 < NX; j2 += NY ) {
                    if (j + j2 < m) {
                        AT[i2 + j2*ldat] = sA[tx + i2][ty + j2];
                    }
                }
            }
        }
        barrier( CLK_LOCAL_MEM_FENCE );
        
        // move to next subtile
        A  += NX;
        AT += NX*ldat;
    }
}


/*
    kernel wrapper to call the device function.
*/
__kernel
void stranspose_kernel(
    magma_int_t m, magma_int_t n,
    __global const float *A, unsigned long A_offset, magma_int_t lda,
    __global float *AT, unsigned long AT_offset,      magma_int_t ldat)
{
    A += A_offset;
    AT += AT_offset;

    stranspose_device(m, n, A, lda, AT, ldat);
}
