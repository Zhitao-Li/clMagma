/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @generated from testing/testing_blas_z.cpp, normal z -> d, Tue Jun 18 16:14:24 2019
       @author Mark Gates
       
       These tests ensure that the MAGMA wrappers around clBLAS calls are
       correct, for example,
       magma_dtrmm(...) and clblasDtrmm(...) produce /exactly/ the same results.
       It tests all combinations of options (trans, uplo, ...) for each size.
*/
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// make sure that asserts are enabled
#undef NDEBUG
#include <assert.h>

// includes, project
#include "testings.h"  // before magma.h, to include clblas_v2
#include "flops.h"
#include "magma.h"
#include "magma_lapack.h"

#define A(i,j)  &A[  (i) + (j)*ld ]
#define C2(i,j) &C2[ (i) + (j)*ld ]
#define LU(i,j) &LU[ (i) + (j)*ld ]

#ifdef HAVE_clBLAS
#define dA(i,j)  dA, ((i) + (j)*ld)
#define dB(i,j)  dB, ((i) + (j)*ld)
#endif

int main( int argc, char** argv )
{
    TESTING_INIT();
    
    real_Double_t   gflops, t1, t2;
    double c_neg_one = MAGMA_D_NEG_ONE;
    magma_int_t ione = 1;
    magma_trans_t trans[] = { MagmaNoTrans, MagmaConjTrans, MagmaTrans };
    magma_uplo_t  uplo [] = { MagmaLower, MagmaUpper };
    magma_diag_t  diag [] = { MagmaUnit, MagmaNonUnit };
    magma_side_t  side [] = { MagmaLeft, MagmaRight };
    
    double  *A,  *B,  *C,   *C2, *LU;
    magmaDouble_ptr dA, dB, dC1, dC2;
    double alpha = MAGMA_D_MAKE( 0.5, 0.1 );
    double beta  = MAGMA_D_MAKE( 0.7, 0.2 );
    double dalpha = 0.6;
    double dbeta  = 0.8;
    double work[1], error, total_error;
    magma_int_t ISEED[4] = {0,0,0,1};
    magma_int_t m, n, k, size, maxn, ld, info;
    magma_int_t *piv;
    magma_int_t err;
    
    magma_opts opts;
    opts.parse_opts( argc, argv );
    //magmablasSetKernelStream( opts.queue );  // opts.handle also uses opts.queue
    
    printf( "Compares magma wrapper function to clBLAS function; all diffs should be exactly 0.\n\n" );
    
    total_error = 0.;
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        m = opts.msize[itest];
        n = opts.nsize[itest];
        k = opts.ksize[itest];
        printf("%%========================================================================\n");
        printf( "m=%d, n=%d, k=%d\n", (int) m, (int) n, (int) k );
        
        // allocate matrices
        // over-allocate so they can be any combination of {m,n,k} x {m,n,k}.
        maxn = max( max( m, n ), k );
        ld = max( 1, maxn );
        size = ld*maxn;
        err = magma_malloc_cpu( (void**) &piv, maxn*sizeof(magma_int_t) );  assert( err == 0 );
        err = magma_dmalloc_cpu( &A,  size );     assert( err == 0 );
        err = magma_dmalloc_cpu( &B,  size );     assert( err == 0 );
        err = magma_dmalloc_cpu( &C,  size );     assert( err == 0 );
        err = magma_dmalloc_cpu( &C2, size );     assert( err == 0 );
        err = magma_dmalloc_cpu( &LU, size );     assert( err == 0 );
        err = magma_dmalloc( &dA,  size );        assert( err == 0 );
        err = magma_dmalloc( &dB,  size );        assert( err == 0 );
        err = magma_dmalloc( &dC1, size );        assert( err == 0 );
        err = magma_dmalloc( &dC2, size );        assert( err == 0 );
        
        // initialize matrices
        size = maxn*maxn;
        lapackf77_dlarnv( &ione, ISEED, &size, A  );
        lapackf77_dlarnv( &ione, ISEED, &size, B  );
        lapackf77_dlarnv( &ione, ISEED, &size, C  );
        
        printf( "%%========= Level 1 BLAS ==========\n" );
        
        // ----- test DSWAP
        // swap columns 2 and 3 of dA, then copy to C2 and compare with A
        if ( n >= 3 ) {
            magma_dsetmatrix( m, n, A, ld, dA, 0, ld, opts.queue );
            magma_dsetmatrix( m, n, A, ld, dB, 0, ld, opts.queue );
            magma_dswap( m, dA(0,1), 1, dA(0,2), 1, opts.queue );
            magma_dswap( m, dB(0,1), 1, dB(0,2), 1, opts.queue );
            
            // check results, storing diff between magma and clblas calls in C2
            clblasDaxpy( ld*n, c_neg_one, dA, 0, 1, dB, 0, 1,
                         1, &opts.queue, 0, NULL, NULL );
            magma_dgetmatrix( m, n, dB, 0, ld, C2, ld, opts.queue );
            error = lapackf77_dlange( "F", &m, &k, C2, &ld, work );
            total_error += error;
            printf( "dswap             diff %.2g\n", error );
        }
        else {
            printf( "dswap skipped for n < 3\n" );
        }
        
        // ----- test IDAMAX
        // get argmax of column of A
        magma_dsetmatrix( m, k, A, ld, dA, 0, ld, opts.queue );
        error = 0;
        for( int j = 0; j < k; ++j ) {
            magma_int_t i1 = magma_idamax( m, dA(0,j), 1, opts.queue );
            magma_int_t i2 = 0;
            // i2 = clblasiDamax( m, dA(0,j), 1,
            //                    1, &opts.queue, 0, NULL, NULL );
            // todo need sync here?
            assert( i1 == i2 );
            error += abs( i1 - i2 );
        }
        total_error += error;
        gflops = (double)m * k / 1e9;
        printf( "idamax            diff %.2g\n", error );
        printf( "\n" );
        
        printf( "%%========= Level 2 BLAS ==========\n" );
        
        // ----- test DGEMV
        // c = alpha*A*b + beta*c,  with A m*n; b,c m or n-vectors
        // try no-trans/trans
        for( int ia = 0; ia < 3; ++ia ) {
            magma_dsetmatrix( m, n, A,  ld, dA, 0, ld, opts.queue );
            magma_dsetvector( maxn, B, 1, dB,  0, 1, opts.queue );
            magma_dsetvector( maxn, C, 1, dC1, 0, 1, opts.queue );
            magma_dsetvector( maxn, C, 1, dC2, 0, 1, opts.queue );
            
            t1 = magma_sync_wtime( opts.queue );
            magma_dgemv( trans[ia], m, n, alpha, dA, 0, ld, dB, 0, 1, beta, dC1, 0, 1, opts.queue );
            t1 = magma_sync_wtime( opts.queue ) - t1;
            
            t2 = magma_sync_wtime( opts.queue );
            clblasDgemv( clblasColumnMajor, clblas_trans_const(trans[ia]),
                         m, n, alpha, dA, 0, ld, dB, 0, 1, beta, dC2, 0, 1,
                         1, &opts.queue, 0, NULL, NULL );
            t2 = magma_sync_wtime( opts.queue ) - t2;
            
            // check results, storing diff between magma and cuda call in C2
            size = (trans[ia] == MagmaNoTrans ? m : n);
            clblasDaxpy( size, c_neg_one, dC1, 0, 1, dC2, 0, 1,
                         1, &opts.queue, 0, NULL, NULL );
            magma_dgetvector( size, dC2, 0, 1, C2, 1, opts.queue );
            error = lapackf77_dlange( "F", &size, &ione, C2, &ld, work );
            total_error += error;
            gflops = FLOPS_DGEMV( m, n ) / 1e9;
            printf( "dgemv( %c )        diff %.2g,  Gflop/s %7.2f, %7.2f\n",
                    lapacke_trans_const(trans[ia]), error, gflops/t1, gflops/t2 );
        }
        printf( "\n" );
        
        // ----- test DSYMV
        // c = alpha*A*b + beta*c,  with A m*m symmetric; b,c m-vectors
        // try upper/lower
        for( int iu = 0; iu < 2; ++iu ) {
            magma_dsetmatrix( m, m, A, ld, dA, 0, ld, opts.queue );
            magma_dsetvector( m, B, 1, dB,  0, 1, opts.queue );
            magma_dsetvector( m, C, 1, dC1, 0, 1, opts.queue );
            magma_dsetvector( m, C, 1, dC2, 0, 1, opts.queue );
            
            t1 = magma_sync_wtime( opts.queue );
            magma_dsymv( uplo[iu], m, alpha, dA, 0, ld, dB, 0, 1, beta, dC1, 0, 1, opts.queue );
            t1 = magma_sync_wtime( opts.queue ) - t1;
            
            t2 = magma_sync_wtime( opts.queue );
            clblasDsymv( clblasColumnMajor, clblas_uplo_const(uplo[iu]),
                         m, alpha, dA, 0, ld, dB, 0, 1, beta, dC2, 0, 1,
                         1, &opts.queue, 0, NULL, NULL );
            t2 = magma_sync_wtime( opts.queue ) - t2;
            
            // check results, storing diff between magma and cuda call in C2
            clblasDaxpy( m, c_neg_one, dC1, 0, 1, dC2, 0, 1,
                         1, &opts.queue, 0, NULL, NULL );
            magma_dgetvector( m, dC2, 0, 1, C2, 1, opts.queue );
            error = lapackf77_dlange( "F", &m, &ione, C2, &ld, work );
            total_error += error;
            gflops = FLOPS_DSYMV( m ) / 1e9;
            printf( "dsymv( %c )        diff %.2g,  Gflop/s %7.2f, %7.2f\n",
                    lapacke_uplo_const(uplo[iu]), error, gflops/t1, gflops/t2 );
        }
        printf( "\n" );
        
        // ----- test DTRSV
        // solve A*c = c,  with A m*m triangular; c m-vector
        // try upper/lower, no-trans/trans, unit/non-unit diag
        // Factor A into LU to get well-conditioned triangles, else solve yields garbage.
        // Still can give garbage if solves aren't consistent with LU factors,
        // e.g., using unit diag for U, so copy lower triangle to upper triangle.
        // Also used for trsm later.
        lapackf77_dlacpy( "Full", &maxn, &maxn, A, &ld, LU, &ld );
        lapackf77_dgetrf( &maxn, &maxn, LU, &ld, piv, &info );
        for( int j = 0; j < maxn; ++j ) {
            for( int i = 0; i < j; ++i ) {
                *LU(i,j) = *LU(j,i);
            }
        }
        for( int iu = 0; iu < 2; ++iu ) {
        for( int it = 0; it < 3; ++it ) {
        for( int id = 0; id < 2; ++id ) {
            magma_dsetmatrix( m, m, LU, ld, dA, 0, ld, opts.queue );
            magma_dsetvector( m, C, 1, dC1, 0, 1, opts.queue );
            magma_dsetvector( m, C, 1, dC2, 0, 1, opts.queue );
            
            t1 = magma_sync_wtime( opts.queue );
            magma_dtrsv( uplo[iu], trans[it], diag[id], m, dA, 0, ld, dC1, 0, 1, opts.queue );
            t1 = magma_sync_wtime( opts.queue ) - t1;
            
            t2 = magma_sync_wtime( opts.queue );
            clblasDtrsv( clblasColumnMajor, clblas_uplo_const(uplo[iu]), clblas_trans_const(trans[it]),
                         clblas_diag_const(diag[id]), m, dA, 0, ld, dC2, 0, 1,
                         1, &opts.queue, 0, NULL, NULL );
            t2 = magma_sync_wtime( opts.queue ) - t2;
            
            // check results, storing diff between magma and cuda call in C2
            clblasDaxpy( m, c_neg_one, dC1, 0, 1, dC2, 0, 1,
                         1, &opts.queue, 0, NULL, NULL );
            magma_dgetvector( m, dC2, 0, 1, C2, 1, opts.queue );
            error = lapackf77_dlange( "F", &m, &ione, C2, &ld, work );
            total_error += error;
            gflops = FLOPS_DTRSM( MagmaLeft, m, 1 ) / 1e9;
            printf( "dtrsv( %c, %c, %c )  diff %.2g,  Gflop/s %7.2f, %7.2f\n",
                    lapacke_uplo_const(uplo[iu]), lapacke_trans_const(trans[it]), lapacke_diag_const(diag[id]),
                    error, gflops/t1, gflops/t2 );
        }}}
        printf( "\n" );
        
        printf( "%%========= Level 3 BLAS ==========\n" );
        
        // ----- test DGEMM
        // C = alpha*A*B + beta*C,  with A m*k or k*m; B k*n or n*k; C m*n
        // try combinations of no-trans/trans
        for( int ia = 0; ia < 3; ++ia ) {
        for( int ib = 0; ib < 3; ++ib ) {
            bool nta = (trans[ia] == MagmaNoTrans);
            bool ntb = (trans[ib] == MagmaNoTrans);
            magma_dsetmatrix( (nta ? m : k), (nta ? m : k), A, ld, dA,  0, ld, opts.queue );
            magma_dsetmatrix( (ntb ? k : n), (ntb ? n : k), B, ld, dB,  0, ld, opts.queue );
            magma_dsetmatrix( m, n, C, ld, dC1, 0, ld, opts.queue );
            magma_dsetmatrix( m, n, C, ld, dC2, 0, ld, opts.queue );
            
            t1 = magma_sync_wtime( opts.queue );
            magma_dgemm( trans[ia], trans[ib], m, n, k, alpha, dA, 0, ld, dB, 0, ld, beta, dC1, 0, ld, opts.queue );
            t1 = magma_sync_wtime( opts.queue ) - t1;
            
            t2 = magma_sync_wtime( opts.queue );
            clblasDgemm( clblasColumnMajor, clblas_trans_const(trans[ia]), clblas_trans_const(trans[ib]),
                         m, n, k, alpha, dA, 0, ld, dB, 0, ld, beta, dC2, 0, ld,
                         1, &opts.queue, 0, NULL, NULL );
            t2 = magma_sync_wtime( opts.queue ) - t2;
            
            // check results, storing diff between magma and cuda call in C2
            clblasDaxpy( ld*n, c_neg_one, dC1, 0, 1, dC2, 0, 1,
                         1, &opts.queue, 0, NULL, NULL );
            magma_dgetmatrix( m, n, dC2, 0, ld, C2, ld, opts.queue );
            error = lapackf77_dlange( "F", &m, &n, C2, &ld, work );
            total_error += error;
            gflops = FLOPS_DGEMM( m, n, k ) / 1e9;
            printf( "dgemm( %c, %c )     diff %.2g,  Gflop/s %7.2f, %7.2f\n",
                    lapacke_trans_const(trans[ia]), lapacke_trans_const(trans[ib]),
                    error, gflops/t1, gflops/t2 );
        }}
        printf( "\n" );
        
        // ----- test DSYMM
        // C = alpha*A*B + beta*C  (left)  with A m*m symmetric; B,C m*n; or
        // C = alpha*B*A + beta*C  (right) with A n*n symmetric; B,C m*n
        // try left/right, upper/lower
        for( int is = 0; is < 2; ++is ) {
        for( int iu = 0; iu < 2; ++iu ) {
            magma_dsetmatrix( m, m, A, ld, dA,  0, ld, opts.queue );
            magma_dsetmatrix( m, n, B, ld, dB,  0, ld, opts.queue );
            magma_dsetmatrix( m, n, C, ld, dC1, 0, ld, opts.queue );
            magma_dsetmatrix( m, n, C, ld, dC2, 0, ld, opts.queue );
            
            t1 = magma_sync_wtime( opts.queue );
            magma_dsymm( side[is], uplo[iu], m, n, alpha, dA, 0, ld, dB, 0, ld, beta, dC1, 0, ld, opts.queue );
            t1 = magma_sync_wtime( opts.queue ) - t1;
            
            t2 = magma_sync_wtime( opts.queue );
            clblasDsymm( clblasColumnMajor, clblas_side_const(side[is]), clblas_uplo_const(uplo[iu]),
                         m, n, alpha, dA, 0, ld, dB, 0, ld, beta, dC2, 0, ld,
                         1, &opts.queue, 0, NULL, NULL );
            t2 = magma_sync_wtime( opts.queue ) - t2;
            
            // check results, storing diff between magma and cuda call in C2
            clblasDaxpy( ld*n, c_neg_one, dC1, 0, 1, dC2, 0, 1,
                         1, &opts.queue, 0, NULL, NULL );
            magma_dgetmatrix( m, n, dC2, 0, ld, C2, ld, opts.queue );
            error = lapackf77_dlange( "F", &m, &n, C2, &ld, work );
            total_error += error;
            gflops = FLOPS_DSYMM( side[is], m, n ) / 1e9;
            printf( "dsymm( %c, %c )     diff %.2g,  Gflop/s %7.2f, %7.2f\n",
                    lapacke_side_const(side[is]), lapacke_uplo_const(uplo[iu]),
                    error, gflops/t1, gflops/t2 );
        }}
        printf( "\n" );
        
        // ----- test DSYRK
        // C = alpha*A*A^H + beta*C  (no-trans) with A m*k and C m*m symmetric; or
        // C = alpha*A^H*A + beta*C  (trans)    with A k*m and C m*m symmetric
        // try upper/lower, no-trans/trans
        for( int iu = 0; iu < 2; ++iu ) {
        for( int it = 0; it < 3; ++it ) {
            magma_dsetmatrix( n, k, A, ld, dA,  0, ld, opts.queue );
            magma_dsetmatrix( n, n, C, ld, dC1, 0, ld, opts.queue );
            magma_dsetmatrix( n, n, C, ld, dC2, 0, ld, opts.queue );
            
            t1 = magma_sync_wtime( opts.queue );
            magma_dsyrk( uplo[iu], trans[it], n, k, dalpha, dA, 0, ld, dbeta, dC1, 0, ld, opts.queue );
            t1 = magma_sync_wtime( opts.queue ) - t1;
            
            t2 = magma_sync_wtime( opts.queue );
            clblasDsyrk( clblasColumnMajor, clblas_uplo_const(uplo[iu]), clblas_trans_const(trans[it]),
                         n, k, dalpha, dA, 0, ld, dbeta, dC2, 0, ld,
                         1, &opts.queue, 0, NULL, NULL );
            t2 = magma_sync_wtime( opts.queue ) - t2;
            
            // check results, storing diff between magma and cuda call in C2
            clblasDaxpy( ld*n, c_neg_one, dC1, 0, 1, dC2, 0, 1,
                         1, &opts.queue, 0, NULL, NULL );
            magma_dgetmatrix( n, n, dC2, 0, ld, C2, ld, opts.queue );
            error = lapackf77_dlange( "F", &n, &n, C2, &ld, work );
            total_error += error;
            gflops = FLOPS_DSYRK( k, n ) / 1e9;
            printf( "dsyrk( %c, %c )     diff %.2g,  Gflop/s %7.2f, %7.2f\n",
                    lapacke_uplo_const(uplo[iu]), lapacke_trans_const(trans[it]),
                    error, gflops/t1, gflops/t2 );
        }}
        printf( "\n" );
        
        // ----- test DSYR2K
        // C = alpha*A*B^H + ^alpha*B*A^H + beta*C  (no-trans) with A,B n*k; C n*n symmetric; or
        // C = alpha*A^H*B + ^alpha*B^H*A + beta*C  (trans)    with A,B k*n; C n*n symmetric
        // try upper/lower, no-trans/trans
        for( int iu = 0; iu < 2; ++iu ) {
        for( int it = 0; it < 3; ++it ) {
            bool nt = (trans[it] == MagmaNoTrans);
            magma_dsetmatrix( (nt ? n : k), (nt ? n : k), A, ld, dA,  0, ld, opts.queue );
            magma_dsetmatrix( n, n, C, ld, dC1, 0, ld, opts.queue );
            magma_dsetmatrix( n, n, C, ld, dC2, 0, ld, opts.queue );
            
            t1 = magma_sync_wtime( opts.queue );
            magma_dsyr2k( uplo[iu], trans[it], n, k, alpha, dA, 0, ld, dB, 0, ld, dbeta, dC1, 0, ld, opts.queue );
            t1 = magma_sync_wtime( opts.queue ) - t1;
            
            t2 = magma_sync_wtime( opts.queue );
            clblasDsyr2k( clblasColumnMajor, clblas_uplo_const(uplo[iu]), clblas_trans_const(trans[it]),
                          n, k, alpha, dA, 0, ld, dB, 0, ld, dbeta, dC2, 0, ld,
                          1, &opts.queue, 0, NULL, NULL );
            t2 = magma_sync_wtime( opts.queue ) - t2;
            
            // check results, storing diff between magma and cuda call in C2
            clblasDaxpy( ld*n, c_neg_one, dC1, 0, 1, dC2, 0, 1,
                         1, &opts.queue, 0, NULL, NULL );
            magma_dgetmatrix( n, n, dC2, 0, ld, C2, ld, opts.queue );
            error = lapackf77_dlange( "F", &n, &n, C2, &ld, work );
            total_error += error;
            gflops = FLOPS_DSYR2K( k, n ) / 1e9;
            printf( "dsyr2k( %c, %c )    diff %.2g,  Gflop/s %7.2f, %7.2f\n",
                    lapacke_uplo_const(uplo[iu]), lapacke_trans_const(trans[it]),
                    error, gflops/t1, gflops/t2 );
        }}
        printf( "\n" );
        
        // ----- test DTRMM
        // C = alpha*A*C  (left)  with A m*m triangular; C m*n; or
        // C = alpha*C*A  (right) with A n*n triangular; C m*n
        // try left/right, upper/lower, no-trans/trans, unit/non-unit
        for( int is = 0; is < 2; ++is ) {
        for( int iu = 0; iu < 2; ++iu ) {
        for( int it = 0; it < 3; ++it ) {
        for( int id = 0; id < 2; ++id ) {
            bool left = (side[is] == MagmaLeft);
            magma_dsetmatrix( (left ? m : n), (left ? m : n), A, ld, dA,  0, ld, opts.queue );
            magma_dsetmatrix( m, n, C, ld, dC1, 0, ld, opts.queue );
            magma_dsetmatrix( m, n, C, ld, dC2, 0, ld, opts.queue );
            
            t1 = magma_sync_wtime( opts.queue );
            magma_dtrmm( side[is], uplo[iu], trans[it], diag[id], m, n, alpha, dA, 0, ld, dC1, 0, ld, opts.queue );
            t1 = magma_sync_wtime( opts.queue ) - t1;
            
            // note clblas does trmm out-of-place (i.e., adds output matrix C),
            // but allows C=B to do in-place.
            t2 = magma_sync_wtime( opts.queue );
            clblasDtrmm( clblasColumnMajor, clblas_side_const(side[is]), clblas_uplo_const(uplo[iu]),
                         clblas_trans_const(trans[it]), clblas_diag_const(diag[id]),
                         m, n, alpha, dA, 0, ld, dC2, 0, ld,
                         1, &opts.queue, 0, NULL, NULL );
            t2 = magma_sync_wtime( opts.queue ) - t2;
            
            // check results, storing diff between magma and cuda call in C2
            clblasDaxpy( ld*n, c_neg_one, dC1, 0, 1, dC2, 0, 1,
                         1, &opts.queue, 0, NULL, NULL );
            magma_dgetmatrix( m, n, dC2, 0, ld, C2, ld, opts.queue );
            error = lapackf77_dlange( "F", &n, &n, C2, &ld, work );
            total_error += error;
            gflops = FLOPS_DTRMM( side[is], m, n ) / 1e9;
            printf( "dtrmm( %c, %c )     diff %.2g,  Gflop/s %7.2f, %7.2f\n",
                    lapacke_uplo_const(uplo[iu]), lapacke_trans_const(trans[it]),
                    error, gflops/t1, gflops/t2 );
        }}}}
        printf( "\n" );
        
        // ----- test DTRSM
        // solve A*X = alpha*B  (left)  with A m*m triangular; B m*n; or
        // solve X*A = alpha*B  (right) with A n*n triangular; B m*n
        // try left/right, upper/lower, no-trans/trans, unit/non-unit
        for( int is = 0; is < 2; ++is ) {
        for( int iu = 0; iu < 2; ++iu ) {
        for( int it = 0; it < 3; ++it ) {
        for( int id = 0; id < 2; ++id ) {
            bool left = (side[is] == MagmaLeft);
            magma_dsetmatrix( (left ? m : n), (left ? m : n), LU, ld, dA,  0, ld, opts.queue );
            magma_dsetmatrix( m, n, C, ld, dC1, 0, ld, opts.queue );
            magma_dsetmatrix( m, n, C, ld, dC2, 0, ld, opts.queue );
            
            t1 = magma_sync_wtime( opts.queue );
            magma_dtrsm( side[is], uplo[iu], trans[it], diag[id], m, n, alpha, dA, 0, ld, dC1, 0, ld, opts.queue );
            t1 = magma_sync_wtime( opts.queue ) - t1;
            
            t2 = magma_sync_wtime( opts.queue );
            clblasDtrsm( clblasColumnMajor, clblas_side_const(side[is]), clblas_uplo_const(uplo[iu]),
                         clblas_trans_const(trans[it]), clblas_diag_const(diag[id]),
                         m, n, alpha, dA, 0, ld, dC2, 0, ld,
                         1, &opts.queue, 0, NULL, NULL );
            t2 = magma_sync_wtime( opts.queue ) - t2;
            
            // check results, storing diff between magma and cuda call in C2
            clblasDaxpy( ld*n, c_neg_one, dC1, 0, 1, dC2, 0, 1,
                         1, &opts.queue, 0, NULL, NULL );
            magma_dgetmatrix( m, n, dC2, 0, ld, C2, ld, opts.queue );
            error = lapackf77_dlange( "F", &n, &n, C2, &ld, work );
            total_error += error;
            gflops = FLOPS_DTRSM( side[is], m, n ) / 1e9;
            printf( "dtrsm( %c, %c )     diff %.2g,  Gflop/s %7.2f, %7.2f\n",
                    lapacke_uplo_const(uplo[iu]), lapacke_trans_const(trans[it]),
                    error, gflops/t1, gflops/t2 );
        }}}}
        printf( "\n" );
        
        // cleanup
        magma_free_cpu( piv );
        magma_free_cpu( A  );
        magma_free_cpu( B  );
        magma_free_cpu( C  );
        magma_free_cpu( C2 );
        magma_free_cpu( LU );
        magma_free( dA  );
        magma_free( dB  );
        magma_free( dC1 );
        magma_free( dC2 );
        fflush( stdout );
    }
    
    if ( total_error != 0. ) {
        printf( "total error %.2g -- ought to be 0 -- some test failed (see above).\n",
                total_error );
    }
    else {
        printf( "all tests passed\n" );
    }
    
    TESTING_FINALIZE();
    
    int status = (total_error != 0.);
    return status;
}
