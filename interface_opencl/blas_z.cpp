/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date
 
       @author Mark Gates
       @author Chongxiao Cao
       @author Stan Tomov
 
       @precisions normal z -> s d c
 */

#include <stdlib.h>
#include <stdio.h>

#include "magma.h"
#include "error.h"

#define COMPLEX
#define PRECISION_z

#if defined(HAVE_clBLAS)


// ========================================
// globals, defined in interface.c
extern magma_event_t* g_event;


// ========================================
// Level 1 BLAS

// --------------------
/** Returns index of element of vector x having max. absolute value;
    i.e., max (infinity) norm.
    
    @param[in]
    n       Number of elements in vector x. n >= 0.
            
    @param[in]
    dx      COMPLEX_16 array on GPU device.
            The n element vector x of dimension (1 + (n-1)*incx).
            
    @param[in]
    incx    Stride between consecutive elements of dx. incx > 0.
            
    @ingroup magma_zblas1
*/
extern "C" magma_int_t
magma_izamax(
    magma_int_t n,
    magmaDoubleComplex_const_ptr dx, size_t dx_offset, magma_int_t incx,
    magma_queue_t queue )
{
    // match reference BLAS edge cases
    if ( n <= 0 || incx <= 0 )
        return 0;
    
//printf( "%s queue %p\n", __func__, queue );
    magmaDoubleComplex hx[10];
    magma_zgetvector( 1, dx, 0, 1, hx, 1, queue );
    
    magma_ptr dimax, scratchBuff;
    magma_int_t e;
//printf( "%s malloc n %d, 2*n+1 %d\n", __func__, n, 2*n+1 );
    e = magma_malloc( &dimax,       sizeof(unsigned int) );
    check_error( e );
    e = magma_malloc( &scratchBuff, (2*n+1)*sizeof(magmaDoubleComplex) );
    check_error( e );
//printf( "diamax %p, scratchBuff %p\n", dimax, scratchBuff );

    unsigned int imax_cpu = 0;
    magma_setvector( 1, sizeof(unsigned int), &imax_cpu, 1, dimax, 0, 1, queue );
    
//printf( "%s clblasiZamax\n", __func__ );
    cl_int err = CLBlastiZamax(
        n, dimax, 0,
        dx, dx_offset, incx,
        &queue, g_event);
    check_error( err );
    
//printf( "%s getvector\n", __func__ );
    magma_getvector( 1, sizeof(unsigned int), dimax, 0, 1, &imax_cpu, 1, queue );
    clFlush(queue);
    
    // work around clBLAS bug: if dx is zero vector it returns 0 instead of 1.
    if ( imax_cpu == 0 )
        imax_cpu = 1;

//printf( "%s free\n", __func__ );
    magma_free( dimax );
    magma_free( scratchBuff );

//printf( "%s done\n", __func__ );
    return imax_cpu;
}

// --------------------
/** Returns the sum of absolute values of vector x; i.e., one norm.

    @param[in]
    n       Number of elements in vector x. n >= 0.

    @param[in]
    dx      COMPLEX_16 array on GPU device.
            The n element vector x of dimension (1 + (n-1)*incx).

    @param[in]
    incx    Stride between consecutive elements of dx. incx > 0.

    @ingroup magma_zblas1
*/
extern "C" double
magma_dzasum(
    magma_int_t n,
    magmaDoubleComplex_const_ptr dx, size_t dx_offset, magma_int_t incx,
    magma_queue_t queue )
{
    assert( false );
    // TODO return clblasDzasum( n, dx, dx_offset, incx );
    return -1;
}

// --------------------
/** Constant times a vector plus a vector; \f$ y = \alpha x + y \f$.

    @param[in]
    n       Number of elements in vectors x and y. n >= 0.

    @param[in]
    alpha   Scalar \f$ \alpha \f$

    @param[in]
    dx      COMPLEX_16 array on GPU device.
            The n element vector x of dimension (1 + (n-1)*incx).

    @param[in]
    incx    Stride between consecutive elements of dx. incx != 0.

    @param[in,out]
    dy      COMPLEX_16 array on GPU device.
            The n element vector y of dimension (1 + (n-1)*incy).

    @param[in]
    incy    Stride between consecutive elements of dy. incy != 0.

    @ingroup magma_zblas1
*/
extern "C" void
magma_zaxpy(
    magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dx, size_t dx_offset, magma_int_t incx,
    magmaDoubleComplex_ptr       dy, size_t dy_offset, magma_int_t incy,
    magma_queue_t queue )
{
    cl_int err = CLBlastZaxpy(
        n, alpha,
        dx, dx_offset, incx,
        dy, dy_offset, incy,
        &queue, g_event );
    check_error( err );
}

// --------------------
/** Copy vector x to vector y; \f$ y = x \f$.

    @param[in]
    n       Number of elements in vectors x and y. n >= 0.

    @param[in]
    dx      COMPLEX_16 array on GPU device.
            The n element vector x of dimension (1 + (n-1)*incx).

    @param[in]
    incx    Stride between consecutive elements of dx. incx != 0.

    @param[out]
    dy      COMPLEX_16 array on GPU device.
            The n element vector y of dimension (1 + (n-1)*incy).

    @param[in]
    incy    Stride between consecutive elements of dy. incy != 0.

    @ingroup magma_zblas1
*/
extern "C" void
magma_zcopy(
    magma_int_t n,
    magmaDoubleComplex_const_ptr dx, size_t dx_offset, magma_int_t incx,
    magmaDoubleComplex_ptr       dy, size_t dy_offset, magma_int_t incy,
    magma_queue_t queue )
{
    if ( n <= 0 )
        return;
    
    cl_int err = CLBlastZcopy( n,
        dx, dx_offset, incx,
        dy, dy_offset, incy,
        &queue, g_event );
    check_error( err );
}

// --------------------
/** Returns dot product of vectors x and y; \f$ x^H y \f$.

    @param[in]
    n       Number of elements in vector x and y. n >= 0.

    @param[in]
    dx      COMPLEX_16 array on GPU device.
            The n element vector x of dimension (1 + (n-1)*incx).

    @param[in]
    incx    Stride between consecutive elements of dx. incx != 0.

    @param[in]
    dy      COMPLEX_16 array on GPU device.
            The n element vector y of dimension (1 + (n-1)*incy).

    @param[in]
    incy    Stride between consecutive elements of dy. incy != 0.

    @ingroup magma_zblas1
*/
extern "C"
magmaDoubleComplex magma_zdotc(
    magma_int_t n,
    magmaDoubleComplex_const_ptr dx, size_t dx_offset, magma_int_t incx,
    magmaDoubleComplex_const_ptr dy, size_t dy_offset, magma_int_t incy,
    magma_queue_t queue )
{
    assert( false );
    // TODO return clblasZdotc(
    // TODO     n,
    // TODO     dx, dx_offset, incx,
    // TODO     dy, dy_offset, incy,
    // TODO     1, &queue, 0, NULL, g_event );
    return MAGMA_Z_ZERO;
}

#ifdef COMPLEX
// --------------------
/** Returns dot product (unconjugated) of vectors x and y; \f$ x^T y \f$.

    @param[in]
    n       Number of elements in vector x and y. n >= 0.

    @param[in]
    dx      COMPLEX_16 array on GPU device.
            The n element vector x of dimension (1 + (n-1)*incx).

    @param[in]
    incx    Stride between consecutive elements of dx. incx != 0.

    @param[in]
    dy      COMPLEX_16 array on GPU device.
            The n element vector y of dimension (1 + (n-1)*incy).

    @param[in]
    incy    Stride between consecutive elements of dy. incy != 0.

    @ingroup magma_zblas1
*/
extern "C"
magmaDoubleComplex magma_zdotu(
    magma_int_t n,
    magmaDoubleComplex_const_ptr dx, size_t dx_offset, magma_int_t incx,
    magmaDoubleComplex_const_ptr dy, size_t dy_offset, magma_int_t incy,
    magma_queue_t queue )
{
    assert( false );
    // TODO return clblasZdotu(
    // TODO     n,
    // TODO     dx, dx_offset, incx,
    // TODO     dy, dy_offset, incy,
    // TODO     1, &queue, 0, NULL, g_event );
    return MAGMA_Z_ZERO;
}
#endif  // COMPLEX

// --------------------
/** Returns 2-norm of vector x. Avoids unnecesary over/underflow.

    @param[in]
    n       Number of elements in vector x and y. n >= 0.

    @param[in]
    dx      COMPLEX_16 array on GPU device.
            The n element vector x of dimension (1 + (n-1)*incx).

    @param[in]
    incx    Stride between consecutive elements of dx. incx > 0.

    @ingroup magma_zblas1
*/
extern "C" double
magma_dznrm2(
    magma_int_t n,
    magmaDoubleComplex_const_ptr dx, size_t dx_offset, magma_int_t incx,
    magma_queue_t queue )
{
    assert( false );
    // TODO return clblasDznrm2(
    // TODO     n, dx, dx_offset, incx,
    // TODO     1, &queue, 0, NULL, g_event );
    return -1;
}

#ifdef REAL
// --------------------
/** Apply Givens plane rotation, where cos (c) is real and sin (s) is complex.

    @param[in]
    n       Number of elements in vector x and y. n >= 0.

    @param[in,out]
    dx      COMPLEX_16 array on GPU device.
            The n element vector x of dimension (1 + (n-1)*incx).
            On output, overwritten with c*x + s*y.

    @param[in]
    incx    Stride between consecutive elements of dx. incx != 0.

    @param[in,out]
    dy      COMPLEX_16 array on GPU device.
            The n element vector y of dimension (1 + (n-1)*incy).
            On output, overwritten with -conj(s)*x + c*y.

    @param[in]
    incy    Stride between consecutive elements of dy. incy != 0.

    @param[in]
    c       double. cosine.

    @param[in]
    s       COMPLEX_16. sine. c and s define a rotation
            [ c         s ]  where c*c + s*conj(s) = 1.
            [ -conj(s)  c ]

    @ingroup magma_zblas1
*/
extern "C" void
magma_zrot(
    magma_int_t n,
    magmaDoubleComplex_ptr dx, size_t dx_offset, magma_int_t incx,
    magmaDoubleComplex_ptr dy, size_t dy_offset, magma_int_t incy,
    double c, magmaDoubleComplex s,
    magma_queue_t queue )
{
    cl_int err = CLBlastZrot(
        n,
        dx, dx_offset, incx,
        dy, dy_offset, incy,
        c, s,
        &queue, g_event );
    check_error( err );
}
#endif // REAL

#ifdef COMPLEX
// --------------------
/** Apply Givens plane rotation, where cos (c) and sin (s) are real.

    @param[in]
    n       Number of elements in vector x and y. n >= 0.

    @param[in,out]
    dx      COMPLEX_16 array on GPU device.
            The n element vector x of dimension (1 + (n-1)*incx).
            On output, overwritten with c*x + s*y.

    @param[in]
    incx    Stride between consecutive elements of dx. incx != 0.

    @param[in,out]
    dy      COMPLEX_16 array on GPU device.
            The n element vector y of dimension (1 + (n-1)*incy).
            On output, overwritten with -conj(s)*x + c*y.

    @param[in]
    incy    Stride between consecutive elements of dy. incy != 0.

    @param[in]
    c       double. cosine.

    @param[in]
    s       double. sine. c and s define a rotation
            [  c  s ]  where c*c + s*s = 1.
            [ -s  c ]

    @ingroup magma_zblas1
*/
extern "C" void
magma_zdrot(
    magma_int_t n,
    magmaDoubleComplex_ptr dx, size_t dx_offset, magma_int_t incx,
    magmaDoubleComplex_ptr dy, size_t dy_offset, magma_int_t incy,
    double c, double s,
    magma_queue_t queue )
{
    cl_int err = CLBlastDrot(
        n,
        dx, dx_offset, incx,
        dy, dy_offset, incy,
        c, s,
        &queue, g_event);
    check_error( err );
}
#endif // COMPLEX

#ifdef REAL
// --------------------
/** Apply modified plane rotation.

    @ingroup magma_zblas1
*/
extern "C" void
magma_zrotm(
    magma_int_t n,
    magmaDouble_ptr dx, size_t dx_offset, magma_int_t incx,
    magmaDouble_ptr dy, size_t dy_offset, magma_int_t incy,
    magmaDouble_const_ptr param, size_t param_offset,
    magma_queue_t queue )
{
    cl_int err = CLBlastZrotm(
        n,
        dx, dx_offset, incx,
        dy, dy_offset, incy,
        param, param_offset,
        &queue, g_event );
    check_error( err );
}

// --------------------
/** Generate modified plane rotation.

    @ingroup magma_zblas1
*/
extern "C" void
magma_zrotmg(
    magmaDouble_ptr       d1, size_t d1_offset,
    magmaDouble_ptr       d2, size_t d2_offset,
    magmaDouble_ptr       x1, size_t x1_offset,
    magmaDouble_const_ptr y1, size_t y1_offset,
    magmaDouble_ptr    param, size_t param_offset,
    magma_queue_t queue )
{
    cl_int err = CLBlastZrotmg(
        d1, d1_offset,
        d2, d2_offset,
        x1, x1_offset,
        y1, y1_offset,
        param, param_offset,
        &queue, g_event );
    check_error( err );
}
#endif // REAL

// --------------------
/** Scales a vector by a constant; \f$ x = \alpha x \f$.

    @param[in]
    n       Number of elements in vector x. n >= 0.

    @param[in]
    alpha   Scalar \f$ \alpha \f$

    @param[in,out]
    dx      COMPLEX_16 array on GPU device.
            The n element vector x of dimension (1 + (n-1)*incx).

    @param[in]
    incx    Stride between consecutive elements of dx. incx > 0.

    @ingroup magma_zblas1
*/
extern "C" void
magma_zscal(
    magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_ptr dx, size_t dx_offset, magma_int_t incx,
    magma_queue_t queue )
{
    if (n <= 0)
        return;

    cl_int err = CLBlastZscal(
        n, alpha, dx, dx_offset, incx,
        &queue, g_event );
    clFlush(queue);
    check_error( err );
}

#ifdef COMPLEX
// --------------------
/** Scales a vector by a real constant; \f$ x = \alpha x \f$.

    @param[in]
    n       Number of elements in vector x. n >= 0.

    @param[in]
    alpha   Scalar \f$ \alpha \f$ (real)

    @param[in,out]
    dx      COMPLEX_16 array on GPU device.
            The n element vector x of dimension (1 + (n-1)*incx).

    @param[in]
    incx    Stride between consecutive elements of dx. incx > 0.

    @ingroup magma_zblas1
*/
extern "C" void
magma_zdscal(
    magma_int_t n,
    double alpha,
    magmaDoubleComplex_ptr dx, size_t dx_offset, magma_int_t incx,
    magma_queue_t queue )
{
    if (n <= 0)
        return;

    cl_int err = CLBlastHscal(
        n, alpha, dx, dx_offset, incx,
        &queue, g_event );
    clFlush(queue);
    check_error( err );
}
#endif // COMPLEX

// --------------------
/** Swap vector x and y; \f$ x <-> y \f$.

    @param[in]
    n       Number of elements in vector x and y. n >= 0.

    @param[in,out]
    dx      COMPLEX_16 array on GPU device.
            The n element vector x of dimension (1 + (n-1)*incx).

    @param[in]
    incx    Stride between consecutive elements of dx. incx != 0.

    @param[in,out]
    dy      COMPLEX_16 array on GPU device.
            The n element vector y of dimension (1 + (n-1)*incy).

    @param[in]
    incy    Stride between consecutive elements of dy. incy != 0.

    @ingroup magma_zblas1
*/
extern "C" void
magma_zswap(
    magma_int_t n,
    magmaDoubleComplex_ptr dx, size_t dx_offset, magma_int_t incx,
    magmaDoubleComplex_ptr dy, size_t dy_offset, magma_int_t incy,
    magma_queue_t queue )
{
    if (n <= 0)
        return;

    cl_int err = CLBlastZswap(
        n, dx, dx_offset, incx,
           dy, dy_offset, incy,
        &queue, g_event );
    clFlush(queue);
    check_error( err );
}


// ========================================
// Level 2 BLAS

// --------------------
/** Perform matrix-vector product.
        \f$ y = \alpha A   x + \beta y \f$  (transA == MagmaNoTrans), or \n
        \f$ y = \alpha A^T x + \beta y \f$  (transA == MagmaTrans),   or \n
        \f$ y = \alpha A^H x + \beta y \f$  (transA == MagmaConjTrans).

    @param[in]
    transA  Operation to perform on A.

    @param[in]
    m       Number of rows of A. m >= 0.

    @param[in]
    n       Number of columns of A. n >= 0.

    @param[in]
    alpha   Scalar \f$ \alpha \f$

    @param[in]
    dA      COMPLEX_16 array of dimension (ldda,n), ldda >= max(1,m).
            The m-by-n matrix A, on GPU device.

    @param[in]
    ldda    Leading dimension of dA.

    @param[in]
    dx      COMPLEX_16 array on GPU device.
            If transA == MagmaNoTrans, the n element vector x of dimension (1 + (n-1)*incx); \n
            otherwise,                 the m element vector x of dimension (1 + (m-1)*incx).

    @param[in]
    incx    Stride between consecutive elements of dx. incx != 0.

    @param[in]
    beta    Scalar \f$ \beta \f$

    @param[in,out]
    dy      COMPLEX_16 array on GPU device.
            If transA == MagmaNoTrans, the m element vector y of dimension (1 + (m-1)*incy); \n
            otherwise,                 the n element vector y of dimension (1 + (n-1)*incy).

    @param[in]
    incy    Stride between consecutive elements of dy. incy != 0.

    @ingroup magma_zblas2
*/
extern "C" void
magma_zgemv(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, size_t dA_offset, magma_int_t ldda,
    magmaDoubleComplex_const_ptr dx, size_t dx_offset, magma_int_t incx,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr       dy, size_t dy_offset, magma_int_t incy,
    magma_queue_t queue )
{
    if ( m <= 0 || n <= 0 )
        return;

    cl_int err = CLBlastZgemv(
        CLBlastLayoutColMajor,
        clblast_trans_const( transA ),
        m, n,
        alpha, dA, dA_offset, ldda,
               dx, dx_offset, incx,
        beta,  dy, dy_offset, incy,
        &queue, g_event );
    clFlush(queue);
    check_error( err );
}

// --------------------
/** Perform rank-1 update, \f$ A = \alpha x y^H + A \f$.

    @param[in]
    m       Number of rows of A. m >= 0.

    @param[in]
    n       Number of columns of A. n >= 0.

    @param[in]
    alpha   Scalar \f$ \alpha \f$

    @param[in]
    dx      COMPLEX_16 array on GPU device.
            The m element vector x of dimension (1 + (m-1)*incx).

    @param[in]
    incx    Stride between consecutive elements of dx. incx != 0.

    @param[in]
    dy      COMPLEX_16 array on GPU device.
            The n element vector y of dimension (1 + (n-1)*incy).

    @param[in]
    incy    Stride between consecutive elements of dy. incy != 0.

    @param[in,out]
    dA      COMPLEX_16 array on GPU device.
            The m-by-n matrix A of dimension (ldda,n), ldda >= max(1,m).

    @param[in]
    ldda    Leading dimension of dA.

    @ingroup magma_zblas2
*/
extern "C" void
magma_zgerc(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dx, size_t dx_offset, magma_int_t incx,
    magmaDoubleComplex_const_ptr dy, size_t dy_offset, magma_int_t incy,
    magmaDoubleComplex_ptr       dA, size_t dA_offset, magma_int_t ldda,
    magma_queue_t queue )
{
    cl_int err = CLBlastZgerc(
        CLBlastLayoutColMajor,
        m, n,
        alpha, dx, dx_offset, incx,
               dy, dy_offset, incy,
               dA, dA_offset, ldda,
        &queue, g_event );
    check_error( err );
}

#ifdef COMPLEX
// --------------------
/** Perform rank-1 update (unconjugated), \f$ A = \alpha x y^H + A \f$.

    @param[in]
    m       Number of rows of A. m >= 0.

    @param[in]
    n       Number of columns of A. n >= 0.

    @param[in]
    alpha   Scalar \f$ \alpha \f$

    @param[in]
    dx      COMPLEX_16 array on GPU device.
            The m element vector x of dimension (1 + (m-1)*incx).

    @param[in]
    incx    Stride between consecutive elements of dx. incx != 0.

    @param[in]
    dy      COMPLEX_16 array on GPU device.
            The n element vector y of dimension (1 + (n-1)*incy).

    @param[in]
    incy    Stride between consecutive elements of dy. incy != 0.

    @param[in,out]
    dA      COMPLEX_16 array of dimension (ldda,n), ldda >= max(1,m).
            The m-by-n matrix A, on GPU device.

    @param[in]
    ldda    Leading dimension of dA.

    @ingroup magma_zblas2
*/
extern "C" void
magma_zgeru(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dx, size_t dx_offset, magma_int_t incx,
    magmaDoubleComplex_const_ptr dy, size_t dy_offset, magma_int_t incy,
    magmaDoubleComplex_ptr       dA, size_t dA_offset, magma_int_t ldda,
    magma_queue_t queue )
{
    cl_int err = CLBlastZgeru(
        CLBlastLayoutColMajor,
        m, n,
        alpha, dx, dx_offset, incx,
               dy, dy_offset, incy,
               dA, dA_offset, ldda,
        &queue, g_event );
    check_error( err );
}
#endif // COMPLEX

// --------------------
/** Perform Hermitian matrix-vector product, \f$ y = \alpha A x + \beta y \f$.

    @param[in]
    uplo    Whether the upper or lower triangle of A is referenced.

    @param[in]
    n       Number of rows and columns of A. n >= 0.

    @param[in]
    alpha   Scalar \f$ \alpha \f$

    @param[in]
    dA      COMPLEX_16 array of dimension (ldda,n), ldda >= max(1,n).
            The n-by-n matrix A, on GPU device.

    @param[in]
    ldda    Leading dimension of dA.

    @param[in]
    dx      COMPLEX_16 array on GPU device.
            The m element vector x of dimension (1 + (m-1)*incx).

    @param[in]
    incx    Stride between consecutive elements of dx. incx != 0.

    @param[in]
    beta    Scalar \f$ \beta \f$

    @param[in,out]
    dy      COMPLEX_16 array on GPU device.
            The n element vector y of dimension (1 + (n-1)*incy).

    @param[in]
    incy    Stride between consecutive elements of dy. incy != 0.

    @ingroup magma_zblas2
*/
extern "C" void
magma_zhemv(
    magma_uplo_t uplo,
    magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, size_t dA_offset, magma_int_t ldda,
    magmaDoubleComplex_const_ptr dx, size_t dx_offset, magma_int_t incx,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr       dy, size_t dy_offset, magma_int_t incy,
    magma_queue_t queue )
{
    if ( n <= 0 )
        return;

#define DEBUG_ZHEMV
#ifdef  DEBUG_ZHEMV
    const magma_int_t ione = 1;
    magmaDoubleComplex *A, *x, *y;
    A = (magmaDoubleComplex*) malloc( ldda * n * sizeof(magmaDoubleComplex) );
    x = (magmaDoubleComplex*) malloc( n * sizeof(magmaDoubleComplex) );
    y = (magmaDoubleComplex*) malloc( n * sizeof(magmaDoubleComplex) );
    assert( A != NULL );
    assert( x != NULL );
    assert( y != NULL );
    magma_zgetmatrix( n, n, dA, dA_offset, ldda, A, ldda, queue );
    magma_zgetvector( n, dx, dx_offset, incx, x, 1, queue );
    magma_zgetvector( n, dy, dy_offset, incy, y, 1, queue );
    blasf77_zhemv( lapack_uplo_const(uplo), &n, &alpha, A, &ldda, x, &ione, &beta, y, &ione );
    magma_zsetvector( n, y, 1, dy, dy_offset, incy, queue );
    free( A );
    free( x );
    free( y );
#else
    cl_int err = clblasZhemv(
        clblasColumnMajor,
        clblas_uplo_const( uplo ),
        n,
        alpha, dA, dA_offset, ldda,
               dx, dx_offset, incx,
        beta,  dy, dy_offset, incy,
        1, &queue, 0, NULL, g_event );
    clFlush(queue);
    check_error( err );
#endif
}

// --------------------
/** Perform Hermitian rank-1 update, \f$ A = \alpha x x^H + A \f$.

    @param[in]
    uplo    Whether the upper or lower triangle of A is referenced.

    @param[in]
    n       Number of rows and columns of A. n >= 0.

    @param[in]
    alpha   Scalar \f$ \alpha \f$

    @param[in]
    dx      COMPLEX_16 array on GPU device.
            The n element vector x of dimension (1 + (n-1)*incx).

    @param[in]
    incx    Stride between consecutive elements of dx. incx != 0.

    @param[in,out]
    dA      COMPLEX_16 array of dimension (ldda,n), ldda >= max(1,n).
            The n-by-n matrix A, on GPU device.

    @param[in]
    ldda    Leading dimension of dA.

    @ingroup magma_zblas2
*/
extern "C" void
magma_zher(
    magma_uplo_t uplo,
    magma_int_t n,
    double alpha,
    magmaDoubleComplex_const_ptr dx, size_t dx_offset, magma_int_t incx,
    magmaDoubleComplex_ptr       dA, size_t dA_offset, magma_int_t ldda,
    magma_queue_t queue )
{
    cl_int err = CLBlastZher(
        CLBlastLayoutColMajor,
        clblast_uplo_const( uplo ),
        n,
        alpha, dx, dx_offset, incx,
               dA, dA_offset, ldda,
        &queue, g_event );
    check_error( err );
}

// --------------------
/** Perform Hermitian rank-2 update, \f$ A = \alpha x y^H + conj(\alpha) y x^H + A \f$.

    @param[in]
    uplo    Whether the upper or lower triangle of A is referenced.

    @param[in]
    n       Number of rows and columns of A. n >= 0.

    @param[in]
    alpha   Scalar \f$ \alpha \f$

    @param[in]
    dx      COMPLEX_16 array on GPU device.
            The n element vector x of dimension (1 + (n-1)*incx).

    @param[in]
    incx    Stride between consecutive elements of dx. incx != 0.

    @param[in]
    dy      COMPLEX_16 array on GPU device.
            The n element vector y of dimension (1 + (n-1)*incy).

    @param[in]
    incy    Stride between consecutive elements of dy. incy != 0.

    @param[in,out]
    dA      COMPLEX_16 array of dimension (ldda,n), ldda >= max(1,n).
            The n-by-n matrix A, on GPU device.

    @param[in]
    ldda    Leading dimension of dA.

    @ingroup magma_zblas2
*/
extern "C" void
magma_zher2(
    magma_uplo_t uplo,
    magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dx, size_t dx_offset, magma_int_t incx,
    magmaDoubleComplex_const_ptr dy, size_t dy_offset, magma_int_t incy,
    magmaDoubleComplex_ptr       dA, size_t dA_offset, magma_int_t ldda,
    magma_queue_t queue )
{
    cl_int err = CLBlastZher2(
        CLBlastLayoutColMajor,
        clblast_uplo_const( uplo ),
        n,
        alpha, dx, dx_offset, incx,
               dy, dy_offset, incy,
               dA, dA_offset, ldda,
        &queue, g_event );
    check_error( err );
}

// --------------------
/** Perform triangular matrix-vector product.
        \f$ x = A   x \f$  (trans == MagmaNoTrans), or \n
        \f$ x = A^T x \f$  (trans == MagmaTrans),   or \n
        \f$ x = A^H x \f$  (trans == MagmaConjTrans).

    @param[in]
    uplo    Whether the upper or lower triangle of A is referenced.

    @param[in]
    trans   Operation to perform on A.

    @param[in]
    diag    Whether the diagonal of A is assumed to be unit or non-unit.

    @param[in]
    n       Number of rows and columns of A. n >= 0.

    @param[in]
    dA      COMPLEX_16 array of dimension (ldda,n), ldda >= max(1,n).
            The n-by-n matrix A, on GPU device.

    @param[in]
    ldda    Leading dimension of dA.

    @param[in]
    dx      COMPLEX_16 array on GPU device.
            The n element vector x of dimension (1 + (n-1)*incx).

    @param[in]
    incx    Stride between consecutive elements of dx. incx != 0.

    @ingroup magma_zblas2
*/
extern "C" void
magma_ztrmv(
    magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t n,
    magmaDoubleComplex_const_ptr dA, size_t dA_offset, magma_int_t ldda,
    magmaDoubleComplex_ptr       dx, size_t dx_offset, magma_int_t incx,
    magma_queue_t queue )
{
    if ( n <= 0 )
        return;

    cl_int err = CLBlastZtrmv(
        CLBlastLayoutColMajor,
        clblast_uplo_const( uplo ),
        clblast_trans_const( trans ),
        clblast_diag_const( diag ),
        n,
        dA, dA_offset, ldda,
        dx, dx_offset, incx,
        &queue, g_event );
    clFlush(queue);
    check_error( err );
}

// --------------------
/** Solve triangular matrix-vector system (one right-hand side).
        \f$ A   x = b \f$  (trans == MagmaNoTrans), or \n
        \f$ A^T x = b \f$  (trans == MagmaTrans),   or \n
        \f$ A^H x = b \f$  (trans == MagmaConjTrans).

    @param[in]
    uplo    Whether the upper or lower triangle of A is referenced.

    @param[in]
    trans   Operation to perform on A.

    @param[in]
    diag    Whether the diagonal of A is assumed to be unit or non-unit.

    @param[in]
    n       Number of rows and columns of A. n >= 0.

    @param[in]
    dA      COMPLEX_16 array of dimension (ldda,n), ldda >= max(1,n).
            The n-by-n matrix A, on GPU device.

    @param[in]
    ldda    Leading dimension of dA.

    @param[in,out]
    dx      COMPLEX_16 array on GPU device.
            On entry, the n element RHS vector b of dimension (1 + (n-1)*incx).
            On exit, overwritten with the solution vector x.

    @param[in]
    incx    Stride between consecutive elements of dx. incx != 0.

    @ingroup magma_zblas2
*/
extern "C" void
magma_ztrsv(
    magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t n,
    magmaDoubleComplex_const_ptr dA, size_t dA_offset, magma_int_t ldda,
    magmaDoubleComplex_ptr       dx, size_t dx_offset, magma_int_t incx,
    magma_queue_t queue )
{
    if ( n <= 0 )
        return;

    cl_int err = CLBlastZtrsv(
        CLBlastLayoutColMajor,
        clblast_uplo_const( uplo ),
        clblast_trans_const( trans ),
        clblast_diag_const( diag ),
        n,
        dA, dA_offset, ldda,
        dx, dx_offset, incx,
        &queue, g_event );
    clFlush(queue);
    check_error( err );
}

// ========================================
// Level 3 BLAS

// --------------------
/** Perform matrix-matrix product, \f$ C = \alpha op(A) op(B) + \beta C \f$.

    @param[in]
    transA  Operation op(A) to perform on matrix A.

    @param[in]
    transB  Operation op(B) to perform on matrix B.

    @param[in]
    m       Number of rows of C and op(A). m >= 0.

    @param[in]
    n       Number of columns of C and op(B). n >= 0.

    @param[in]
    k       Number of columns of op(A) and rows of op(B). k >= 0.

    @param[in]
    alpha   Scalar \f$ \alpha \f$

    @param[in]
    dA      COMPLEX_16 array on GPU device.
            If transA == MagmaNoTrans, the m-by-k matrix A of dimension (ldda,k), ldda >= max(1,m); \n
            otherwise,                 the k-by-m matrix A of dimension (ldda,m), ldda >= max(1,k).

    @param[in]
    ldda    Leading dimension of dA.

    @param[in]
    dB      COMPLEX_16 array on GPU device.
            If transB == MagmaNoTrans, the k-by-n matrix B of dimension (lddb,n), lddb >= max(1,k); \n
            otherwise,                 the n-by-k matrix B of dimension (lddb,k), lddb >= max(1,n).

    @param[in]
    lddb    Leading dimension of dB.

    @param[in]
    beta    Scalar \f$ \beta \f$

    @param[in,out]
    dC      COMPLEX_16 array on GPU device.
            The m-by-n matrix C of dimension (lddc,n), lddc >= max(1,m).

    @param[in]
    lddc    Leading dimension of dC.

    @ingroup magma_zblas3
*/
extern "C" void
magma_zgemm(
    magma_trans_t transA, magma_trans_t transB,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, size_t dA_offset, magma_int_t ldda,
    magmaDoubleComplex_const_ptr dB, size_t dB_offset, magma_int_t lddb,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr       dC, size_t dC_offset, magma_int_t lddc,
    magma_queue_t queue )
{
    if ( m <= 0 || n <= 0 || k <= 0 )
        return;

    cl_int err = CLBlastZgemm(
        CLBlastLayoutColMajor,
        clblast_trans_const( transA ),
        clblast_trans_const( transB ),
        m, n, k,
        alpha, dA, dA_offset, ldda,
               dB, dB_offset, lddb,
        beta,  dC, dC_offset, lddc,
        &queue, g_event );
    clFlush(queue);
    check_error( err );
}

// --------------------
/** Perform symmetric matrix-matrix product.
        \f$ C = \alpha A B + \beta C \f$ (side == MagmaLeft), or \n
        \f$ C = \alpha B A + \beta C \f$ (side == MagmaRight),   \n
        where \f$ A \f$ is symmetric.

    @param[in]
    side    Whether A is on the left or right.

    @param[in]
    uplo    Whether the upper or lower triangle of A is referenced.

    @param[in]
    m       Number of rows of C. m >= 0.

    @param[in]
    n       Number of columns of C. n >= 0.

    @param[in]
    alpha   Scalar \f$ \alpha \f$

    @param[in]
    dA      COMPLEX_16 array on GPU device.
            If side == MagmaLeft, the m-by-m symmetric matrix A of dimension (ldda,m), ldda >= max(1,m); \n
            otherwise,            the n-by-n symmetric matrix A of dimension (ldda,n), ldda >= max(1,n).

    @param[in]
    ldda    Leading dimension of dA.

    @param[in]
    dB      COMPLEX_16 array on GPU device.
            The m-by-n matrix B of dimension (lddb,n), lddb >= max(1,m).

    @param[in]
    lddb    Leading dimension of dB.

    @param[in]
    beta    Scalar \f$ \beta \f$

    @param[in,out]
    dC      COMPLEX_16 array on GPU device.
            The m-by-n matrix C of dimension (lddc,n), lddc >= max(1,m).

    @param[in]
    lddc    Leading dimension of dC.

    @ingroup magma_zblas3
*/
extern "C" void
magma_zsymm(
    magma_side_t side, magma_uplo_t uplo,
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, size_t dA_offset, magma_int_t ldda,
    magmaDoubleComplex_const_ptr dB, size_t dB_offset, magma_int_t lddb,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr       dC, size_t dC_offset, magma_int_t lddc,
    magma_queue_t queue )
{
    cl_int err = CLBlastZsymm(
        CLBlastLayoutColMajor,
        clblast_side_const( side ),
        clblast_uplo_const( uplo ),
        m, n,
        alpha, dA, dA_offset, ldda,
               dB, dB_offset, lddb,
        beta,  dC, dC_offset, lddc,
        &queue, g_event );
    check_error( err );
}

// --------------------
/** Perform symmetric rank-k update.
        \f$ C = \alpha A A^T + \beta C \f$ (trans == MagmaNoTrans), or \n
        \f$ C = \alpha A^T A + \beta C \f$ (trans == MagmaTrans),      \n
        where \f$ C \f$ is symmetric.

    @param[in]
    uplo    Whether the upper or lower triangle of C is referenced.

    @param[in]
    trans   Operation to perform on A.

    @param[in]
    n       Number of rows and columns of C. n >= 0.

    @param[in]
    k       Number of columns of A (for MagmaNoTrans) or rows of A (for MagmaTrans). k >= 0.

    @param[in]
    alpha   Scalar \f$ \alpha \f$

    @param[in]
    dA      COMPLEX_16 array on GPU device.
            If trans == MagmaNoTrans, the n-by-k matrix A of dimension (ldda,k), ldda >= max(1,n); \n
            otherwise,                the k-by-n matrix A of dimension (ldda,n), ldda >= max(1,k).

    @param[in]
    ldda    Leading dimension of dA.

    @param[in]
    beta    Scalar \f$ \beta \f$

    @param[in,out]
    dC      COMPLEX_16 array on GPU device.
            The n-by-n symmetric matrix C of dimension (lddc,n), lddc >= max(1,n).

    @param[in]
    lddc    Leading dimension of dC.

    @ingroup magma_zblas3
*/
extern "C" void
magma_zsyrk(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, size_t dA_offset, magma_int_t ldda,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr       dC, size_t dC_offset, magma_int_t lddc,
    magma_queue_t queue )
{
    if (n <= 0 || k <= 0)
        return;

    cl_int err = CLBlastZsyrk(
        CLBlastLayoutColMajor,
        clblast_uplo_const( uplo ),
        clblast_trans_const( trans ),
        n, k,
        alpha, dA, dA_offset, ldda,
        beta,  dC, dC_offset, lddc,
        &queue, g_event );
    check_error( err );
}

// --------------------
/** Perform symmetric rank-2k update.
        \f$ C = \alpha A B^T + \alpha B A^T \beta C \f$ (trans == MagmaNoTrans), or \n
        \f$ C = \alpha A^T B + \alpha B^T A \beta C \f$ (trans == MagmaTrans),      \n
        where \f$ C \f$ is symmetric.

    @param[in]
    uplo    Whether the upper or lower triangle of C is referenced.

    @param[in]
    trans   Operation to perform on A and B.

    @param[in]
    n       Number of rows and columns of C. n >= 0.

    @param[in]
    k       Number of columns of A and B (for MagmaNoTrans) or rows of A and B (for MagmaTrans). k >= 0.

    @param[in]
    alpha   Scalar \f$ \alpha \f$

    @param[in]
    dA      COMPLEX_16 array on GPU device.
            If trans == MagmaNoTrans, the n-by-k matrix A of dimension (ldda,k), ldda >= max(1,n); \n
            otherwise,                the k-by-n matrix A of dimension (ldda,n), ldda >= max(1,k).

    @param[in]
    ldda    Leading dimension of dA.

    @param[in]
    dB      COMPLEX_16 array on GPU device.
            If trans == MagmaNoTrans, the n-by-k matrix B of dimension (lddb,k), lddb >= max(1,n); \n
            otherwise,                the k-by-n matrix B of dimension (lddb,n), lddb >= max(1,k).

    @param[in]
    lddb    Leading dimension of dB.

    @param[in]
    beta    Scalar \f$ \beta \f$

    @param[in,out]
    dC      COMPLEX_16 array on GPU device.
            The n-by-n symmetric matrix C of dimension (lddc,n), lddc >= max(1,n).

    @param[in]
    lddc    Leading dimension of dC.

    @ingroup magma_zblas3
*/
extern "C" void
magma_zsyr2k(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, size_t dA_offset, magma_int_t ldda,
    magmaDoubleComplex_const_ptr dB, size_t dB_offset, magma_int_t lddb,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr       dC, size_t dC_offset, magma_int_t lddc,
    magma_queue_t queue )
{
    cl_int err = CLBlastZsyr2k(
        CLBlastLayoutColMajor,
        clblast_uplo_const( uplo ),
        clblast_trans_const( trans ),
        n, k,
        alpha, dA, dA_offset, ldda,
               dB, dB_offset, lddb,
        beta,  dC, dC_offset, lddc,
        &queue, g_event );
    check_error( err );
}

#ifdef COMPLEX
// --------------------
/** Perform Hermitian matrix-matrix product.
        \f$ C = \alpha A B + \beta C \f$ (side == MagmaLeft), or \n
        \f$ C = \alpha B A + \beta C \f$ (side == MagmaRight),   \n
        where \f$ A \f$ is Hermitian.

    @param[in]
    side    Whether A is on the left or right.

    @param[in]
    uplo    Whether the upper or lower triangle of A is referenced.

    @param[in]
    m       Number of rows of C. m >= 0.

    @param[in]
    n       Number of columns of C. n >= 0.

    @param[in]
    alpha   Scalar \f$ \alpha \f$

    @param[in]
    dA      COMPLEX_16 array on GPU device.
            If side == MagmaLeft, the m-by-m Hermitian matrix A of dimension (ldda,m), ldda >= max(1,m); \n
            otherwise,            the n-by-n Hermitian matrix A of dimension (ldda,n), ldda >= max(1,n).

    @param[in]
    ldda    Leading dimension of dA.

    @param[in]
    dB      COMPLEX_16 array on GPU device.
            The m-by-n matrix B of dimension (lddb,n), lddb >= max(1,m).

    @param[in]
    lddb    Leading dimension of dB.

    @param[in]
    beta    Scalar \f$ \beta \f$

    @param[in,out]
    dC      COMPLEX_16 array on GPU device.
            The m-by-n matrix C of dimension (lddc,n), lddc >= max(1,m).

    @param[in]
    lddc    Leading dimension of dC.

    @ingroup magma_zblas3
*/
extern "C" void
magma_zhemm(
    magma_side_t side, magma_uplo_t uplo,
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, size_t dA_offset, magma_int_t ldda,
    magmaDoubleComplex_const_ptr dB, size_t dB_offset, magma_int_t lddb,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr       dC, size_t dC_offset, magma_int_t lddc,
    magma_queue_t queue )
{
    if ( m <= 0 || n <= 0)
        return;

    cl_int err = CLBlastZhemm(
        CLBlastLayoutColMajor,
        clblast_side_const( side ),
        clblast_uplo_const( uplo ),
        m, n,
        alpha, dA, dA_offset, ldda,
               dB, dB_offset, lddb,
        beta,  dC, dC_offset, lddc,
        &queue, g_event );
    clFlush(queue);
    check_error( err );
}

// --------------------
/** Perform Hermitian rank-k update.
        \f$ C = \alpha A A^T + \beta C \f$ (trans == MagmaNoTrans), or \n
        \f$ C = \alpha A^T A + \beta C \f$ (trans == MagmaTrans),      \n
        where \f$ C \f$ is Hermitian.

    @param[in]
    uplo    Whether the upper or lower triangle of C is referenced.

    @param[in]
    trans   Operation to perform on A.

    @param[in]
    n       Number of rows and columns of C. n >= 0.

    @param[in]
    k       Number of columns of A (for MagmaNoTrans) or rows of A (for MagmaTrans). k >= 0.

    @param[in]
    alpha   Scalar \f$ \alpha \f$

    @param[in]
    dA      COMPLEX_16 array on GPU device.
            If trans == MagmaNoTrans, the n-by-k matrix A of dimension (ldda,k), ldda >= max(1,n); \n
            otherwise,                the k-by-n matrix A of dimension (ldda,n), ldda >= max(1,k).

    @param[in]
    ldda    Leading dimension of dA.

    @param[in]
    beta    Scalar \f$ \beta \f$

    @param[in,out]
    dC      COMPLEX_16 array on GPU device.
            The n-by-n Hermitian matrix C of dimension (lddc,n), lddc >= max(1,n).

    @param[in]
    lddc    Leading dimension of dC.

    @ingroup magma_zblas3
*/
extern "C" void
magma_zherk(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    double alpha,
    magmaDoubleComplex_const_ptr dA, size_t dA_offset, magma_int_t ldda,
    double beta,
    magmaDoubleComplex_ptr       dC, size_t dC_offset, magma_int_t lddc,
    magma_queue_t queue )
{
    if (n <= 0 || k <= 0)
        return;

    cl_int err = CLBlastZherk(
        CLBlastLayoutColMajor,
        clblast_uplo_const( uplo ),
        clblast_trans_const( trans ),
        n, k,
        alpha, dA, dA_offset, ldda,
        beta,  dC, dC_offset, lddc,
        &queue, g_event );
    clFlush(queue);
    check_error( err );
}

// --------------------
/** Perform Hermitian rank-2k update.
        \f$ C = \alpha A B^T + \alpha B A^T \beta C \f$ (trans == MagmaNoTrans), or \n
        \f$ C = \alpha A^T B + \alpha B^T A \beta C \f$ (trans == MagmaTrans),      \n
        where \f$ C \f$ is Hermitian.

    @param[in]
    uplo    Whether the upper or lower triangle of C is referenced.

    @param[in]
    trans   Operation to perform on A and B.

    @param[in]
    n       Number of rows and columns of C. n >= 0.

    @param[in]
    k       Number of columns of A and B (for MagmaNoTrans) or rows of A and B (for MagmaTrans). k >= 0.

    @param[in]
    alpha   Scalar \f$ \alpha \f$

    @param[in]
    dA      COMPLEX_16 array on GPU device.
            If trans == MagmaNoTrans, the n-by-k matrix A of dimension (ldda,k), ldda >= max(1,n); \n
            otherwise,                the k-by-n matrix A of dimension (ldda,n), ldda >= max(1,k).

    @param[in]
    ldda    Leading dimension of dA.

    @param[in]
    dB      COMPLEX_16 array on GPU device.
            If trans == MagmaNoTrans, the n-by-k matrix B of dimension (lddb,k), lddb >= max(1,n); \n
            otherwise,                the k-by-n matrix B of dimension (lddb,n), lddb >= max(1,k).

    @param[in]
    lddb    Leading dimension of dB.

    @param[in]
    beta    Scalar \f$ \beta \f$

    @param[in,out]
    dC      COMPLEX_16 array on GPU device.
            The n-by-n Hermitian matrix C of dimension (lddc,n), lddc >= max(1,n).

    @param[in]
    lddc    Leading dimension of dC.

    @ingroup magma_zblas3
*/
extern "C" void
magma_zher2k(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, size_t dA_offset, magma_int_t ldda,
    magmaDoubleComplex_const_ptr dB, size_t dB_offset, magma_int_t lddb,
    double beta,
    magmaDoubleComplex_ptr dC, size_t dC_offset, magma_int_t lddc,
    magma_queue_t queue )
{
    if (n <= 0 || k <= 0)
        return;

    cl_int err = CLBlastZher2k(
        CLBlastLayoutColMajor,
        clblast_uplo_const( uplo ),
        clblast_trans_const( trans ),
        n, k,
        alpha, dA, dA_offset, ldda,
        dB, dB_offset, lddb,
        beta, dC, dC_offset, lddc,
        &queue, g_event );
    clFlush(queue);
    check_error( err );
}
#endif // COMPLEX

// --------------------
/** Perform triangular matrix-matrix product.
        \f$ B = \alpha op(A) B \f$ (side == MagmaLeft), or \n
        \f$ B = \alpha B op(A) \f$ (side == MagmaRight),   \n
        where \f$ A \f$ is triangular.

    @param[in]
    side    Whether A is on the left or right.

    @param[in]
    uplo    Whether A is upper or lower triangular.

    @param[in]
    trans   Operation to perform on A.

    @param[in]
    diag    Whether the diagonal of A is assumed to be unit or non-unit.

    @param[in]
    m       Number of rows of B. m >= 0.

    @param[in]
    n       Number of columns of B. n >= 0.

    @param[in]
    alpha   Scalar \f$ \alpha \f$

    @param[in]
    dA      COMPLEX_16 array on GPU device.
            If side == MagmaLeft, the n-by-n triangular matrix A of dimension (ldda,n), ldda >= max(1,n); \n
            otherwise,            the m-by-m triangular matrix A of dimension (ldda,m), ldda >= max(1,m).

    @param[in]
    ldda    Leading dimension of dA.

    @param[in]
    dB      COMPLEX_16 array on GPU device.
            The m-by-n matrix B of dimension (lddb,n), lddb >= max(1,m).

    @param[in]
    lddb    Leading dimension of dB.

    @ingroup magma_zblas3
*/
extern "C" void
magma_ztrmm(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, size_t dA_offset, magma_int_t ldda,
    magmaDoubleComplex_ptr       dB, size_t dB_offset, magma_int_t lddb,
    magma_queue_t queue )
{
    if (m <= 0 || n <= 0)
        return;
    
//#define DEBUG_TRMM
//#ifdef  DEBUG_TRMM
//#else
//#endif

#ifdef PRECISION_z
if ( side == MagmaLeft && trans == MagmaNoTrans ) {
printf( "ztrmm( %c, %c, %c, %c, %d, %d, %d, %d )\n",
        lapacke_side_const(side),
        lapacke_uplo_const(uplo),
        lapacke_trans_const(trans),
        lapacke_diag_const(diag),
        m, n, ldda, lddb );
    magmaDoubleComplex *A, *B;
    int k = (side == MagmaLeft ? m : n);
    A = (magmaDoubleComplex*) malloc( ldda * k * sizeof(magmaDoubleComplex) );
    B = (magmaDoubleComplex*) malloc( lddb * n * sizeof(magmaDoubleComplex) );
    assert( A != NULL );
    assert( B != NULL );
    magma_zgetmatrix( k, k, dA, dA_offset, ldda, A, ldda, queue );
    magma_zgetmatrix( m, n, dB, dB_offset, lddb, B, lddb, queue );
    blasf77_ztrmm( lapack_side_const(side), lapack_uplo_const(uplo),
                   lapack_trans_const(trans), lapack_diag_const(diag),
                   &m, &n, &alpha, A, &ldda, B, &lddb );
    magma_zsetmatrix( m, n, B, lddb, dB, dB_offset, lddb, queue );
    free( A );
    free( B );
printf( "ztrmm done\n" );
}
else {
#endif
    cl_int err = CLBlastZtrmm(
        CLBlastLayoutColMajor,
        clblast_side_const( side ),
        clblast_uplo_const( uplo ),
        clblast_trans_const( trans ),
        clblast_diag_const( diag ),
        m, n,
        alpha, dA, dA_offset, ldda,
               dB, dB_offset, lddb,
        &queue, g_event );
    clFlush(queue);
    check_error( err );
#ifdef PRECISION_z
}
#endif
}

// --------------------
/** Solve triangular matrix-matrix system (multiple right-hand sides).
        \f$ op(A) X = \alpha B \f$ (side == MagmaLeft), or \n
        \f$ X op(A) = \alpha B \f$ (side == MagmaRight),   \n
        where \f$ A \f$ is triangular.

    @param[in]
    side    Whether A is on the left or right.

    @param[in]
    uplo    Whether A is upper or lower triangular.

    @param[in]
    trans   Operation to perform on A.

    @param[in]
    diag    Whether the diagonal of A is assumed to be unit or non-unit.

    @param[in]
    m       Number of rows of B. m >= 0.

    @param[in]
    n       Number of columns of B. n >= 0.

    @param[in]
    alpha   Scalar \f$ \alpha \f$

    @param[in]
    dA      COMPLEX_16 array on GPU device.
            If side == MagmaLeft, the m-by-m triangular matrix A of dimension (ldda,m), ldda >= max(1,m); \n
            otherwise,            the n-by-n triangular matrix A of dimension (ldda,n), ldda >= max(1,n).

    @param[in]
    ldda    Leading dimension of dA.

    @param[in,out]
    dB      COMPLEX_16 array on GPU device.
            On entry, m-by-n matrix B of dimension (lddb,n), lddb >= max(1,m).
            On exit, overwritten with the solution matrix X.

    @param[in]
    lddb    Leading dimension of dB.

    @ingroup magma_zblas3
*/
extern "C" void
magma_ztrsm(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, size_t dA_offset, magma_int_t ldda,
    magmaDoubleComplex_ptr       dB, size_t dB_offset, magma_int_t lddb,
    magma_queue_t queue )
{
    if (m <= 0 || n <= 0)
        return;

#define DEBUG_TRSM
#ifdef  DEBUG_TRSM
printf( "ztrsm( %c, %c, %c, %c, %d, %d, %d, %d )\n",
        lapacke_side_const(side),
        lapacke_uplo_const(uplo),
        lapacke_trans_const(trans),
        lapacke_diag_const(diag),
        m, n, ldda, lddb );
    magmaDoubleComplex *A, *B;
    int k = (side == MagmaLeft ? m : n);
    A = (magmaDoubleComplex*) malloc( ldda * k * sizeof(magmaDoubleComplex) );
    B = (magmaDoubleComplex*) malloc( lddb * n * sizeof(magmaDoubleComplex) );
    assert( A != NULL );
    assert( B != NULL );
//printf( "get dA( %d, %d, %p, %lu, %d, %p, %d, %p )\n", k, k, dA, dA_offset, ldda, A, ldda, queue );
    magma_zgetmatrix( k, k, dA, dA_offset, ldda, A, ldda, queue );
//printf( "get dB( %d, %d, %p, %lu, %d, %p, %d, %p )\n", m, n, dB, dB_offset, lddb, B, lddb, queue );
    magma_zgetmatrix( m, n, dB, dB_offset, lddb, B, lddb, queue );
//printf( "blasf77_ztrsm\n" );
    blasf77_ztrsm( lapack_side_const(side), lapack_uplo_const(uplo),
                   lapack_trans_const(trans), lapack_diag_const(diag),
                   &m, &n, &alpha, A, &ldda, B, &lddb );
//printf( "set dB( %d, %d, %p, %d, %p, %lu, %d, %p )\n", m, n, B, lddb, dB, dB_offset, lddb, queue );
    magma_zsetmatrix( m, n, B, lddb, dB, dB_offset, lddb, queue );
    free( A );
    free( B );
printf( "ztrsm done\n" );
#else
    cl_int err = CLBlastZtrsm(
        CLBlastLayoutColMajor,
        clblast_side_const( side ),
        clblast_uplo_const( uplo ),
        clblast_trans_const( trans ),
        clblast_diag_const( diag ),
        m, n,
        alpha, dA, dA_offset, ldda,
               dB, dB_offset, lddb,
        &queue, g_event );
    clFlush(queue);
    check_error( err );
#endif
}

#endif // HAVE_clblas

#undef COMPLEX
