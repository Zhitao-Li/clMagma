/*
    -- clMAGMA (version 1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date
       
       @author Stan Tomov
       @author Mark Gates
       
       @generated from clmagmablas/zbcyclic.cpp, normal z -> c, Tue Jun 18 16:14:18 2019
*/
#include "common_magma.h"

#define PRECISION_c

#ifdef HAVE_clBLAS
    #define dA( dev, i_, j_ )  dA[dev], ((i_) + (j_)*ldda)
#else
    #define dA( dev, i_, j_ ) (dA[dev] + (i_) + (j_)*ldda)
#endif

#define hA( i_, j_ ) (hA + (i_) + (j_)*lda)


//===========================================================================
// Set a matrix from CPU to multi-GPUs in 1D column block cyclic distribution.
// The dA arrays are pointers to the matrix data on the corresponding GPUs.
//===========================================================================
extern "C" void
magma_csetmatrix_1D_col_bcyclic(
    magma_int_t m, magma_int_t n,
    const magmaFloatComplex *hA, magma_int_t lda,
    magmaFloatComplex_ptr   *dA, magma_int_t ldda,
    magma_int_t ngpu, magma_int_t nb,
    magma_queue_t* queues )
{
    magma_int_t info = 0;
    if ( m < 0 )
        info = -1;
    else if ( n < 0 )
        info = -2;
    else if ( lda < m )
        info = -4;
    else if ( ldda < m )
        info = -6;
    else if ( ngpu < 1 )
        info = -7;
    else if ( nb < 1 )
        info = -8;
    
    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;  //info;
    }
    
    magma_int_t j, dev, jb;
    
    //magma_device_t cdevice;
    //magma_getdevice( &cdevice );
    
    for( j = 0; j < n; j += nb ) {
        dev = (j/nb) % ngpu;
        //magma_setdevice( dev );
        jb = min(nb, n-j);
        magma_csetmatrix_async( m, jb,
                                hA(0,j), lda,
                                dA( dev, 0, j/(nb*ngpu)*nb ), ldda,
                                queues[dev], NULL );
    }
    
    //magma_setdevice( cdevice );
}


//===========================================================================
// Get a matrix with 1D column block cyclic distribution from multi-GPUs to the CPU.
// The dA arrays are pointers to the matrix data on the corresponding GPUs.
//===========================================================================
extern "C" void
magma_cgetmatrix_1D_col_bcyclic(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex_const_ptr const *dA, magma_int_t ldda,
    magmaFloatComplex                 *hA, magma_int_t lda,
    magma_int_t ngpu, magma_int_t nb,
    magma_queue_t* queues )
{
    magma_int_t info = 0;
    if ( m < 0 )
        info = -1;
    else if ( n < 0 )
        info = -2;
    else if ( ldda < m )
        info = -4;
    else if ( lda < m )
        info = -6;
    else if ( ngpu < 1 )
        info = -7;
    else if ( nb < 1 )
        info = -8;
    
    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;  //info;
    }
    
    magma_int_t j, dev, jb;
    
    //magma_device_t cdevice;
    //magma_getdevice( &cdevice );
    
    for( j = 0; j < n; j += nb ) {
        dev = (j/nb) % ngpu;
        //magma_setdevice( dev );
        jb = min(nb, n-j);
        magma_cgetmatrix_async( m, jb,
                                dA( dev, 0, j/(nb*ngpu)*nb ), ldda,
                                hA(0,j), lda,
                                queues[dev], NULL );
    }
    
    //magma_setdevice( cdevice );
}


//===========================================================================
// Set a matrix from CPU to multi-GPUs in 1D row block cyclic distribution.
// The dA arrays are pointers to the matrix data on the corresponding GPUs.
//===========================================================================
extern "C" void
magma_csetmatrix_1D_row_bcyclic(
    magma_int_t m, magma_int_t n,
    const magmaFloatComplex    *hA, magma_int_t lda,
    magmaFloatComplex_ptr      *dA, magma_int_t ldda,
    magma_int_t ngpu, magma_int_t nb,
    magma_queue_t* queues )
{
    magma_int_t info = 0;
    if ( m < 0 )
        info = -1;
    else if ( n < 0 )
        info = -2;
    else if ( lda < m )
        info = -4;
    else if ( ldda < (1+m/(nb*ngpu))*nb )
        info = -6;
    else if ( ngpu < 1 )
        info = -7;
    else if ( nb < 1 )
        info = -8;
    
    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;  //info;
    }
    
    magma_int_t i, dev, jb;
    
    //magma_device_t cdevice;
    //magma_getdevice( &cdevice );
    
    for( i = 0; i < m; i += nb ) {
        dev = (i/nb) % ngpu;
        //magma_setdevice( dev );
        jb = min(nb, m-i);
        magma_csetmatrix_async( jb, n,
                                hA(i,0), lda,
                                dA( dev, i/(nb*ngpu)*nb, 0 ), ldda,
                                queues[dev], NULL );
    }
    
    //magma_setdevice( cdevice );
}


//===========================================================================
// Get a matrix with 1D row block cyclic distribution from multi-GPUs to the CPU.
// The dA arrays are pointers to the matrix data for the corresponding GPUs.
//===========================================================================
extern "C" void
magma_cgetmatrix_1D_row_bcyclic(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex_const_ptr const *dA, magma_int_t ldda,
    magmaFloatComplex                 *hA, magma_int_t lda,
    magma_int_t ngpu, magma_int_t nb,
    magma_queue_t* queues )
{
    magma_int_t info = 0;
    if ( m < 0 )
        info = -1;
    else if ( n < 0 )
        info = -2;
    else if ( ldda < (1+m/(nb*ngpu))*nb )
        info = -4;
    else if ( lda < m )
        info = -6;
    else if ( ngpu < 1 )
        info = -7;
    else if ( nb < 1 )
        info = -8;
    
    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;  //info;
    }
    
    magma_int_t i, dev, jb;
    
    //magma_device_t cdevice;
    //magma_getdevice( &cdevice );
    
    for( i = 0; i < m; i += nb ) {
        dev = (i/nb) % ngpu;
        //magma_setdevice( dev );
        jb = min(nb, m-i);
        magma_cgetmatrix_async( jb, n,
                                dA( dev, i/(nb*ngpu)*nb, 0 ), ldda,
                                hA(i,0), lda,
                                queues[dev], NULL );
    }
    
    //magma_setdevice( cdevice );
}
