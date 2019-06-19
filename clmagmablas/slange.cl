/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @generated from clmagmablas/zlange.cl, normal z -> s, Tue Jun 18 16:14:14 2019

       auto-converted from slange.cu
       @author Mark Gates
*/
#include "kernels_header.h"
#include "slange.h"
#include "magma_smax_nan.h"
#include "reduce.h"

/* Computes row sums dwork[i] = sum( abs( A(i,:) )), i=0:m-1, for || A ||_inf,
 * where m and n are any size.
 * Has ceil( m/NB_X ) blocks of NB_X threads. Each thread does one row.
 * See also slange_max_kernel code, below. */
__kernel void
slange_inf_kernel(
    magma_int_t m, magma_int_t n,
    __global const float *A, unsigned long A_offset, magma_int_t lda,
    __global float *dwork, unsigned long dwork_offset )
{
    A += A_offset;
    dwork += dwork_offset;

    int i = get_group_id(0)*NB_X + get_local_id(0);
    float rsum[4] = {0, 0, 0, 0};
    int n_mod_4 = n % 4;
    n -= n_mod_4;
    
    // if beyond last row, skip row
    if ( i < m ) {
        A += i;
        
        if ( n >= 4 ) {
            __global const float *Aend = A + lda*n;
            float rA[4] = { A[0], A[lda], A[2*lda], A[3*lda] };
            A += 4*lda;
            
            while( A < Aend ) {
                rsum[0] += MAGMA_S_ABS( rA[0] );  rA[0] = A[0];
                rsum[1] += MAGMA_S_ABS( rA[1] );  rA[1] = A[lda];
                rsum[2] += MAGMA_S_ABS( rA[2] );  rA[2] = A[2*lda];
                rsum[3] += MAGMA_S_ABS( rA[3] );  rA[3] = A[3*lda];
                A += 4*lda;
            }
            
            rsum[0] += MAGMA_S_ABS( rA[0] );
            rsum[1] += MAGMA_S_ABS( rA[1] );
            rsum[2] += MAGMA_S_ABS( rA[2] );
            rsum[3] += MAGMA_S_ABS( rA[3] );
        }
    
        /* clean up code */
        switch( n_mod_4 ) {
            case 0:
                break;
    
            case 1:
                rsum[0] += MAGMA_S_ABS( A[0] );
                break;
    
            case 2:
                rsum[0] += MAGMA_S_ABS( A[0]   );
                rsum[1] += MAGMA_S_ABS( A[lda] );
                break;
    
            case 3:
                rsum[0] += MAGMA_S_ABS( A[0]     );
                rsum[1] += MAGMA_S_ABS( A[lda]   );
                rsum[2] += MAGMA_S_ABS( A[2*lda] );
                break;
        }
    
        /* compute final result */
        dwork[i] = rsum[0] + rsum[1] + rsum[2] + rsum[3];
    }
}


/* Computes max of row dwork[i] = max( abs( A(i,:) )), i=0:m-1, for || A ||_max,
 * where m and n are any size.
 * Has ceil( m/NB_X ) blocks of NB_X threads. Each thread does one row.
 * Based on slange_inf_kernel code, above. */
__kernel void
slange_max_kernel(
    magma_int_t m, magma_int_t n,
    __global const float *A, unsigned long A_offset, magma_int_t lda,
    __global float *dwork, unsigned long dwork_offset )
{
    A += A_offset;
    dwork += dwork_offset;

    int i = get_group_id(0)*NB_X + get_local_id(0);
    float rmax[4] = {0, 0, 0, 0};
    int n_mod_4 = n % 4;
    n -= n_mod_4;
    
    // if beyond last row, skip row
    if ( i < m ) {
        A += i;
        
        if ( n >= 4 ) {
            __global const float *Aend = A + lda*n;
            float rA[4] = { A[0], A[lda], A[2*lda], A[3*lda] };
            A += 4*lda;
            
            while( A < Aend ) {
                rmax[0] = max_nan( rmax[0], MAGMA_S_ABS( rA[0] ));  rA[0] = A[0];
                rmax[1] = max_nan( rmax[1], MAGMA_S_ABS( rA[1] ));  rA[1] = A[lda];
                rmax[2] = max_nan( rmax[2], MAGMA_S_ABS( rA[2] ));  rA[2] = A[2*lda];
                rmax[3] = max_nan( rmax[3], MAGMA_S_ABS( rA[3] ));  rA[3] = A[3*lda];
                A += 4*lda;
            }
            
            rmax[0] = max_nan( rmax[0], MAGMA_S_ABS( rA[0] ));
            rmax[1] = max_nan( rmax[1], MAGMA_S_ABS( rA[1] ));
            rmax[2] = max_nan( rmax[2], MAGMA_S_ABS( rA[2] ));
            rmax[3] = max_nan( rmax[3], MAGMA_S_ABS( rA[3] ));
        }
    
        /* clean up code */
        switch( n_mod_4 ) {
            case 0:
                break;
    
            case 1:
                rmax[0] = max_nan( rmax[0], MAGMA_S_ABS( A[0] ));
                break;                          
                                                
            case 2:                             
                rmax[0] = max_nan( rmax[0], MAGMA_S_ABS( A[  0] ));
                rmax[1] = max_nan( rmax[1], MAGMA_S_ABS( A[lda] ));
                break;                          
                                                
            case 3:                             
                rmax[0] = max_nan( rmax[0], MAGMA_S_ABS( A[    0] ));
                rmax[1] = max_nan( rmax[1], MAGMA_S_ABS( A[  lda] ));
                rmax[2] = max_nan( rmax[2], MAGMA_S_ABS( A[2*lda] ));
                break;
        }
    
        /* compute final result */
        dwork[i] = max_nan( max_nan( max_nan( rmax[0], rmax[1] ), rmax[2] ), rmax[3] );
    }
}


/* Computes col sums dwork[j] = sum( abs( A(:,j) )), j=0:n-1, for || A ||_one,
 * where m and n are any size.
 * Has n blocks of NB threads each. Block j sums one column, A(:,j) into dwork[j].
 * Thread i accumulates A(i,j) + A(i+NB,j) + A(i+2*NB,j) + ... into ssum[i],
 * then threads collectively do a sum-reduction of ssum,
 * and finally thread 0 saves to dwork[j]. */
__kernel void
slange_one_kernel(
    magma_int_t m, magma_int_t n,
    __global const float *A, unsigned long A_offset, magma_int_t lda,
    __global float *dwork, unsigned long dwork_offset )
{
    A += A_offset;
    dwork += dwork_offset;

    __local float ssum[NB_X];
    int tx = get_local_id(0);
    
    A += get_group_id(0)*lda;  // column j
    
    ssum[tx] = 0;
    for( int i = tx; i < m; i += NB_X ) {
        ssum[tx] += MAGMA_S_ABS( A[i] );
    }
    magma_ssum_reduce( NB_X, tx, ssum );
    if ( tx == 0 ) {
        dwork[ get_group_id(0) ] = ssum[0];
    }
}
