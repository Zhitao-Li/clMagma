/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c

       auto-converted from zlange.cu
       @author Mark Gates
*/
#include "kernels_header.h"
#include "zlange.h"
#include "magma_dmax_nan.h"
#include "reduce.h"

/* Computes row sums dwork[i] = sum( abs( A(i,:) )), i=0:m-1, for || A ||_inf,
 * where m and n are any size.
 * Has ceil( m/NB_X ) blocks of NB_X threads. Each thread does one row.
 * See also zlange_max_kernel code, below. */
__kernel void
zlange_inf_kernel(
    magma_int_t m, magma_int_t n,
    __global const magmaDoubleComplex *A, unsigned long A_offset, magma_int_t lda,
    __global double *dwork, unsigned long dwork_offset )
{
    A += A_offset;
    dwork += dwork_offset;

    int i = get_group_id(0)*NB_X + get_local_id(0);
    double rsum[4] = {0, 0, 0, 0};
    int n_mod_4 = n % 4;
    n -= n_mod_4;
    
    // if beyond last row, skip row
    if ( i < m ) {
        A += i;
        
        if ( n >= 4 ) {
            __global const magmaDoubleComplex *Aend = A + lda*n;
            magmaDoubleComplex rA[4] = { A[0], A[lda], A[2*lda], A[3*lda] };
            A += 4*lda;
            
            while( A < Aend ) {
                rsum[0] += MAGMA_Z_ABS( rA[0] );  rA[0] = A[0];
                rsum[1] += MAGMA_Z_ABS( rA[1] );  rA[1] = A[lda];
                rsum[2] += MAGMA_Z_ABS( rA[2] );  rA[2] = A[2*lda];
                rsum[3] += MAGMA_Z_ABS( rA[3] );  rA[3] = A[3*lda];
                A += 4*lda;
            }
            
            rsum[0] += MAGMA_Z_ABS( rA[0] );
            rsum[1] += MAGMA_Z_ABS( rA[1] );
            rsum[2] += MAGMA_Z_ABS( rA[2] );
            rsum[3] += MAGMA_Z_ABS( rA[3] );
        }
    
        /* clean up code */
        switch( n_mod_4 ) {
            case 0:
                break;
    
            case 1:
                rsum[0] += MAGMA_Z_ABS( A[0] );
                break;
    
            case 2:
                rsum[0] += MAGMA_Z_ABS( A[0]   );
                rsum[1] += MAGMA_Z_ABS( A[lda] );
                break;
    
            case 3:
                rsum[0] += MAGMA_Z_ABS( A[0]     );
                rsum[1] += MAGMA_Z_ABS( A[lda]   );
                rsum[2] += MAGMA_Z_ABS( A[2*lda] );
                break;
        }
    
        /* compute final result */
        dwork[i] = rsum[0] + rsum[1] + rsum[2] + rsum[3];
    }
}


/* Computes max of row dwork[i] = max( abs( A(i,:) )), i=0:m-1, for || A ||_max,
 * where m and n are any size.
 * Has ceil( m/NB_X ) blocks of NB_X threads. Each thread does one row.
 * Based on zlange_inf_kernel code, above. */
__kernel void
zlange_max_kernel(
    magma_int_t m, magma_int_t n,
    __global const magmaDoubleComplex *A, unsigned long A_offset, magma_int_t lda,
    __global double *dwork, unsigned long dwork_offset )
{
    A += A_offset;
    dwork += dwork_offset;

    int i = get_group_id(0)*NB_X + get_local_id(0);
    double rmax[4] = {0, 0, 0, 0};
    int n_mod_4 = n % 4;
    n -= n_mod_4;
    
    // if beyond last row, skip row
    if ( i < m ) {
        A += i;
        
        if ( n >= 4 ) {
            __global const magmaDoubleComplex *Aend = A + lda*n;
            magmaDoubleComplex rA[4] = { A[0], A[lda], A[2*lda], A[3*lda] };
            A += 4*lda;
            
            while( A < Aend ) {
                rmax[0] = max_nan( rmax[0], MAGMA_Z_ABS( rA[0] ));  rA[0] = A[0];
                rmax[1] = max_nan( rmax[1], MAGMA_Z_ABS( rA[1] ));  rA[1] = A[lda];
                rmax[2] = max_nan( rmax[2], MAGMA_Z_ABS( rA[2] ));  rA[2] = A[2*lda];
                rmax[3] = max_nan( rmax[3], MAGMA_Z_ABS( rA[3] ));  rA[3] = A[3*lda];
                A += 4*lda;
            }
            
            rmax[0] = max_nan( rmax[0], MAGMA_Z_ABS( rA[0] ));
            rmax[1] = max_nan( rmax[1], MAGMA_Z_ABS( rA[1] ));
            rmax[2] = max_nan( rmax[2], MAGMA_Z_ABS( rA[2] ));
            rmax[3] = max_nan( rmax[3], MAGMA_Z_ABS( rA[3] ));
        }
    
        /* clean up code */
        switch( n_mod_4 ) {
            case 0:
                break;
    
            case 1:
                rmax[0] = max_nan( rmax[0], MAGMA_Z_ABS( A[0] ));
                break;                          
                                                
            case 2:                             
                rmax[0] = max_nan( rmax[0], MAGMA_Z_ABS( A[  0] ));
                rmax[1] = max_nan( rmax[1], MAGMA_Z_ABS( A[lda] ));
                break;                          
                                                
            case 3:                             
                rmax[0] = max_nan( rmax[0], MAGMA_Z_ABS( A[    0] ));
                rmax[1] = max_nan( rmax[1], MAGMA_Z_ABS( A[  lda] ));
                rmax[2] = max_nan( rmax[2], MAGMA_Z_ABS( A[2*lda] ));
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
zlange_one_kernel(
    magma_int_t m, magma_int_t n,
    __global const magmaDoubleComplex *A, unsigned long A_offset, magma_int_t lda,
    __global double *dwork, unsigned long dwork_offset )
{
    A += A_offset;
    dwork += dwork_offset;

    __local double ssum[NB_X];
    int tx = get_local_id(0);
    
    A += get_group_id(0)*lda;  // column j
    
    ssum[tx] = 0;
    for( int i = tx; i < m; i += NB_X ) {
        ssum[tx] += MAGMA_Z_ABS( A[i] );
    }
    magma_dsum_reduce( NB_X, tx, ssum );
    if ( tx == 0 ) {
        dwork[ get_group_id(0) ] = ssum[0];
    }
}
