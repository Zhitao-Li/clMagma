/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date
 
       @author Mark Gates
       @author Chongxiao Cao
       @author Stan Tomov
 
       @generated from interface_opencl/blas_z.cpp, normal z -> s, Tue Jun 18 16:14:15 2019
 */

#include <stdlib.h>
#include <stdio.h>

#include "magma.h"
#include "error.h"

#define REAL
#define PRECISION_s

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
    dx      REAL array on GPU device.
            The n element vector x of dimension (1 + (n-1)*incx).
            
    @param[in]
    incx    Stride between consecutive elements of dx. incx > 0.
            
    @ingroup magma_sblas1
*/
extern "C" magma_int_t
magma_isamax(
    magma_int_t n,
    magmaFloat_const_ptr dx, size_t dx_offset, magma_int_t incx,
    magma_queue_t queue )
{
    // match reference BLAS edge cases
    if ( n <= 0 || incx <= 0 )
        return 0;
    
//printf( "%s queue %p\n", __func__, queue );
    float hx[10];
    magma_sgetvector( 1, dx, 0, 1, hx, 1, queue );
    
    magma_ptr dimax, scratchBuff;
    magma_int_t e;
//printf( "%s malloc n %d, 2*n+1 %d\n", __func__, n, 2*n+1 );
    e = magma_malloc( &dimax,       sizeof(unsigned int) );
    check_error( e );
    e = magma_malloc( &scratchBuff, (2*n+1)*sizeof(float) );
    check_error( e );
//printf( "diamax %p, scratchBuff %p\n", dimax, scratchBuff );

    unsigned int imax_cpu = 0;
    magma_setvector( 1, sizeof(unsigned int), &imax_cpu, 1, dimax, 0, 1, queue );
    
//printf( "%s clblasiSamax\n", __func__ );
    cl_int err = clblasiSamax(
        n, dimax, 0,
        dx, dx_offset, incx,
        scratchBuff,
        1, &queue, 0, NULL, g_event);
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
    dx      REAL array on GPU device.
            The n element vector x of dimension (1 + (n-1)*incx).

    @param[in]
    incx    Stride between consecutive elements of dx. incx > 0.

    @ingroup magma_sblas1
*/
extern "C" float
magma_sasum(
    magma_int_t n,
    magmaFloat_const_ptr dx, size_t dx_offset, magma_int_t incx,
    magma_queue_t queue )
{
    assert( false );
    // TODO return clblasSasum( n, dx, dx_offset, incx );
    return -1;
}

// --------------------
/** Constant times a vector plus a vector; \f$ y = \alpha x + y \f$.

    @param[in]
    n       Number of elements in vectors x and y. n >= 0.

    @param[in]
    alpha   Scalar \f$ \alpha \f$

    @param[in]
    dx      REAL array on GPU device.
            The n element vector x of dimension (1 + (n-1)*incx).

    @param[in]
    incx    Stride between consecutive elements of dx. incx != 0.

    @param[in,out]
    dy      REAL array on GPU device.
            The n element vector y of dimension (1 + (n-1)*incy).

    @param[in]
    incy    Stride between consecutive elements of dy. incy != 0.

    @ingroup magma_sblas1
*/
extern "C" void
magma_saxpy(
    magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dx, size_t dx_offset, magma_int_t incx,
    magmaFloat_ptr       dy, size_t dy_offset, magma_int_t incy,
    magma_queue_t queue )
{
    cl_int err = clblasSaxpy(
        n, alpha,
        dx, dx_offset, incx,
        dy, dy_offset, incy,
        1, &queue, 0, NULL, g_event );
    check_error( err );
}

// --------------------
/** Copy vector x to vector y; \f$ y = x \f$.

    @param[in]
    n       Number of elements in vectors x and y. n >= 0.

    @param[in]
    dx      REAL array on GPU device.
            The n element vector x of dimension (1 + (n-1)*incx).

    @param[in]
    incx    Stride between consecutive elements of dx. incx != 0.

    @param[out]
    dy      REAL array on GPU device.
            The n element vector y of dimension (1 + (n-1)*incy).

    @param[in]
    incy    Stride between consecutive elements of dy. incy != 0.

    @ingroup magma_sblas1
*/
extern "C" void
magma_scopy(
    magma_int_t n,
    magmaFloat_const_ptr dx, size_t dx_offset, magma_int_t incx,
    magmaFloat_ptr       dy, size_t dy_offset, magma_int_t incy,
    magma_queue_t queue )
{
    if ( n <= 0 )
        return;
    
    cl_int err = clblasScopy( n,
        dx, dx_offset, incx,
        dy, dy_offset, incy,
        1, &queue, 0, NULL, g_event );
    check_error( err );
}

// --------------------
/** Returns dot product of vectors x and y; \f$ x^H y \f$.

    @param[in]
    n       Number of elements in vector x and y. n >= 0.

    @param[in]
    dx      REAL array on GPU device.
            The n element vector x of dimension (1 + (n-1)*incx).

    @param[in]
    incx    Stride between consecutive elements of dx. incx != 0.

    @param[in]
    dy      REAL array on GPU device.
            The n element vector y of dimension (1 + (n-1)*incy).

    @param[in]
    incy    Stride between consecutive elements of dy. incy != 0.

    @ingroup magma_sblas1
*/
extern "C"
float magma_sdot(
    magma_int_t n,
    magmaFloat_const_ptr dx, size_t dx_offset, magma_int_t incx,
    magmaFloat_const_ptr dy, size_t dy_offset, magma_int_t incy,
    magma_queue_t queue )
{
    assert( false );
    // TODO return clblasSdot(
    // TODO     n,
    // TODO     dx, dx_offset, incx,
    // TODO     dy, dy_offset, incy,
    // TODO     1, &queue, 0, NULL, g_event );
    return MAGMA_S_ZERO;
}

#ifdef COMPLEX
// --------------------
/** Returns dot product (unconjugated) of vectors x and y; \f$ x^T y \f$.

    @param[in]
    n       Number of elements in vector x and y. n >= 0.

    @param[in]
    dx      REAL array on GPU device.
            The n element vector x of dimension (1 + (n-1)*incx).

    @param[in]
    incx    Stride between consecutive elements of dx. incx != 0.

    @param[in]
    dy      REAL array on GPU device.
            The n element vector y of dimension (1 + (n-1)*incy).

    @param[in]
    incy    Stride between consecutive elements of dy. incy != 0.

    @ingroup magma_sblas1
*/
extern "C"
float magma_sdot(
    magma_int_t n,
    magmaFloat_const_ptr dx, size_t dx_offset, magma_int_t incx,
    magmaFloat_const_ptr dy, size_t dy_offset, magma_int_t incy,
    magma_queue_t queue )
{
    assert( false );
    // TODO return clblasSdot(
    // TODO     n,
    // TODO     dx, dx_offset, incx,
    // TODO     dy, dy_offset, incy,
    // TODO     1, &queue, 0, NULL, g_event );
    return MAGMA_S_ZERO;
}
#endif  // COMPLEX

// --------------------
/** Returns 2-norm of vector x. Avoids unnecesary over/underflow.

    @param[in]
    n       Number of elements in vector x and y. n >= 0.

    @param[in]
    dx      REAL array on GPU device.
            The n element vector x of dimension (1 + (n-1)*incx).

    @param[in]
    incx    Stride between consecutive elements of dx. incx > 0.

    @ingroup magma_sblas1
*/
extern "C" float
magma_snrm2(
    magma_int_t n,
    magmaFloat_const_ptr dx, size_t dx_offset, magma_int_t incx,
    magma_queue_t queue )
{
    assert( false );
    // TODO return clblasSnrm2(
    // TODO     n, dx, dx_offset, incx,
    // TODO     1, &queue, 0, NULL, g_event );
    return -1;
}

#ifdef REAL
// --------------------
/** Apply Givens plane rotation, where cos (c) is real and sin (s) is real.

    @param[in]
    n       Number of elements in vector x and y. n >= 0.

    @param[in,out]
    dx      REAL array on GPU device.
            The n element vector x of dimension (1 + (n-1)*incx).
            On output, overwritten with c*x + s*y.

    @param[in]
    incx    Stride between consecutive elements of dx. incx != 0.

    @param[in,out]
    dy      REAL array on GPU device.
            The n element vector y of dimension (1 + (n-1)*incy).
            On output, overwritten with -conj(s)*x + c*y.

    @param[in]
    incy    Stride between consecutive elements of dy. incy != 0.

    @param[in]
    c       float. cosine.

    @param[in]
    s       REAL. sine. c and s define a rotation
            [ c         s ]  where c*c + s*conj(s) = 1.
            [ -conj(s)  c ]

    @ingroup magma_sblas1
*/
extern "C" void
magma_srot(
    magma_int_t n,
    magmaFloat_ptr dx, size_t dx_offset, magma_int_t incx,
    magmaFloat_ptr dy, size_t dy_offset, magma_int_t incy,
    float c, float s,
    magma_queue_t queue )
{
    cl_int err = clblasSrot(
        n,
        dx, dx_offset, incx,
        dy, dy_offset, incy,
        c, s,
        1, &queue, 0, NULL, g_event );
    check_error( err );
}
#endif // REAL

#ifdef COMPLEX
// --------------------
/** Apply Givens plane rotation, where cos (c) and sin (s) are real.

    @param[in]
    n       Number of elements in vector x and y. n >= 0.

    @param[in,out]
    dx      REAL array on GPU device.
            The n element vector x of dimension (1 + (n-1)*incx).
            On output, overwritten with c*x + s*y.

    @param[in]
    incx    Stride between consecutive elements of dx. incx != 0.

    @param[in,out]
    dy      REAL array on GPU device.
            The n element vector y of dimension (1 + (n-1)*incy).
            On output, overwritten with -conj(s)*x + c*y.

    @param[in]
    incy    Stride between consecutive elements of dy. incy != 0.

    @param[in]
    c       float. cosine.

    @param[in]
    s       float. sine. c and s define a rotation
            [  c  s ]  where c*c + s*s = 1.
            [ -s  c ]

    @ingroup magma_sblas1
*/
extern "C" void
magma_srot(
    magma_int_t n,
    magmaFloat_ptr dx, size_t dx_offset, magma_int_t incx,
    magmaFloat_ptr dy, size_t dy_offset, magma_int_t incy,
    float c, float s,
    magma_queue_t queue )
{
    cl_int err = clblasSrot(
        n,
        dx, dx_offset, incx,
        dy, dy_offset, incy,
        c, s,
        1, &queue, 0, NULL, g_event);
    check_error( err );
}
#endif // COMPLEX

#ifdef REAL
// --------------------
/** Apply modified plane rotation.

    @ingroup magma_sblas1
*/
extern "C" void
magma_srotm(
    magma_int_t n,
    magmaFloat_ptr dx, size_t dx_offset, magma_int_t incx,
    magmaFloat_ptr dy, size_t dy_offset, magma_int_t incy,
    magmaFloat_const_ptr param, size_t param_offset,
    magma_queue_t queue )
{
    cl_int err = clblasSrotm(
        n,
        dx, dx_offset, incx,
        dy, dy_offset, incy,
        param, param_offset,
        1, &queue, 0, NULL, g_event );
    check_error( err );
}

// --------------------
/** Generate modified plane rotation.

    @ingroup magma_sblas1
*/
extern "C" void
magma_srotmg(
    magmaFloat_ptr       d1, size_t d1_offset,
    magmaFloat_ptr       d2, size_t d2_offset,
    magmaFloat_ptr       x1, size_t x1_offset,
    magmaFloat_const_ptr y1, size_t y1_offset,
    magmaFloat_ptr    param, size_t param_offset,
    magma_queue_t queue )
{
    cl_int err = clblasSrotmg(
        d1, d1_offset,
        d2, d2_offset,
        x1, x1_offset,
        y1, y1_offset,
        param, param_offset,
        1, &queue, 0, NULL, g_event );
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
    dx      REAL array on GPU device.
            The n element vector x of dimension (1 + (n-1)*incx).

    @param[in]
    incx    Stride between consecutive elements of dx. incx > 0.

    @ingroup magma_sblas1
*/
extern "C" void
magma_sscal(
    magma_int_t n,
    float alpha,
    magmaFloat_ptr dx, size_t dx_offset, magma_int_t incx,
    magma_queue_t queue )
{
    if (n <= 0)
        return;

    cl_int err = clblasSscal(
        n, alpha, dx, dx_offset, incx,
        1, &queue, 0, NULL, g_event );
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
    dx      REAL array on GPU device.
            The n element vector x of dimension (1 + (n-1)*incx).

    @param[in]
    incx    Stride between consecutive elements of dx. incx > 0.

    @ingroup magma_sblas1
*/
extern "C" void
magma_sscal(
    magma_int_t n,
    float alpha,
    magmaFloat_ptr dx, size_t dx_offset, magma_int_t incx,
    magma_queue_t queue )
{
    if (n <= 0)
        return;

    cl_int err = clblasSscal(
        n, alpha, dx, dx_offset, incx,
        1, &queue, 0, NULL, g_event );
    clFlush(queue);
    check_error( err );
}
#endif // COMPLEX

// --------------------
/** Swap vector x and y; \f$ x <-> y \f$.

    @param[in]
    n       Number of elements in vector x and y. n >= 0.

    @param[in,out]
    dx      REAL array on GPU device.
            The n element vector x of dimension (1 + (n-1)*incx).

    @param[in]
    incx    Stride between consecutive elements of dx. incx != 0.

    @param[in,out]
    dy      REAL array on GPU device.
            The n element vector y of dimension (1 + (n-1)*incy).

    @param[in]
    incy    Stride between consecutive elements of dy. incy != 0.

    @ingroup magma_sblas1
*/
extern "C" void
magma_sswap(
    magma_int_t n,
    magmaFloat_ptr dx, size_t dx_offset, magma_int_t incx,
    magmaFloat_ptr dy, size_t dy_offset, magma_int_t incy,
    magma_queue_t queue )
{
    if (n <= 0)
        return;

    cl_int err = clblasSswap(
        n, dx, dx_offset, incx,
           dy, dy_offset, incy,
        1, &queue, 0, NULL, g_event );
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
    dA      REAL array of dimension (ldda,n), ldda >= max(1,m).
            The m-by-n matrix A, on GPU device.

    @param[in]
    ldda    Leading dimension of dA.

    @param[in]
    dx      REAL array on GPU device.
            If transA == MagmaNoTrans, the n element vector x of dimension (1 + (n-1)*incx); \n
            otherwise,                 the m element vector x of dimension (1 + (m-1)*incx).

    @param[in]
    incx    Stride between consecutive elements of dx. incx != 0.

    @param[in]
    beta    Scalar \f$ \beta \f$

    @param[in,out]
    dy      REAL array on GPU device.
            If transA == MagmaNoTrans, the m element vector y of dimension (1 + (m-1)*incy); \n
            otherwise,                 the n element vector y of dimension (1 + (n-1)*incy).

    @param[in]
    incy    Stride between consecutive elements of dy. incy != 0.

    @ingroup magma_sblas2
*/
extern "C" void
magma_sgemv(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dA, size_t dA_offset, magma_int_t ldda,
    magmaFloat_const_ptr dx, size_t dx_offset, magma_int_t incx,
    float beta,
    magmaFloat_ptr       dy, size_t dy_offset, magma_int_t incy,
    magma_queue_t queue )
{
    if ( m <= 0 || n <= 0 )
        return;

    cl_int err = clblasSgemv(
        clblasColumnMajor,
        clblas_trans_const( transA ),
        m, n,
        alpha, dA, dA_offset, ldda,
               dx, dx_offset, incx,
        beta,  dy, dy_offset, incy,
        1, &queue, 0, NULL, g_event );
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
    dx      REAL array on GPU device.
            The m element vector x of dimension (1 + (m-1)*incx).

    @param[in]
    incx    Stride between consecutive elements of dx. incx != 0.

    @param[in]
    dy      REAL array on GPU device.
            The n element vector y of dimension (1 + (n-1)*incy).

    @param[in]
    incy    Stride between consecutive elements of dy. incy != 0.

    @param[in,out]
    dA      REAL array on GPU device.
            The m-by-n matrix A of dimension (ldda,n), ldda >= max(1,m).

    @param[in]
    ldda    Leading dimension of dA.

    @ingroup magma_sblas2
*/
extern "C" void
magma_sger(
    magma_int_t m, magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dx, size_t dx_offset, magma_int_t incx,
    magmaFloat_const_ptr dy, size_t dy_offset, magma_int_t incy,
    magmaFloat_ptr       dA, size_t dA_offset, magma_int_t ldda,
    magma_queue_t queue )
{
    cl_int err = clblasSger(
        clblasColumnMajor,
        m, n,
        alpha, dx, dx_offset, incx,
               dy, dy_offset, incy,
               dA, dA_offset, ldda,
        1, &queue, 0, NULL, g_event );
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
    dx      REAL array on GPU device.
            The m element vector x of dimension (1 + (m-1)*incx).

    @param[in]
    incx    Stride between consecutive elements of dx. incx != 0.

    @param[in]
    dy      REAL array on GPU device.
            The n element vector y of dimension (1 + (n-1)*incy).

    @param[in]
    incy    Stride between consecutive elements of dy. incy != 0.

    @param[in,out]
    dA      REAL array of dimension (ldda,n), ldda >= max(1,m).
            The m-by-n matrix A, on GPU device.

    @param[in]
    ldda    Leading dimension of dA.

    @ingroup magma_sblas2
*/
extern "C" void
magma_sger(
    magma_int_t m, magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dx, size_t dx_offset, magma_int_t incx,
    magmaFloat_const_ptr dy, size_t dy_offset, magma_int_t incy,
    magmaFloat_ptr       dA, size_t dA_offset, magma_int_t ldda,
    magma_queue_t queue )
{
    cl_int err = clblasSger(
        clblasColumnMajor,
        m, n,
        alpha, dx, dx_offset, incx,
               dy, dy_offset, incy,
               dA, dA_offset, ldda,
        1, &queue, 0, NULL, g_event );
    check_error( err );
}
#endif // COMPLEX

// --------------------
/** Perform symmetric matrix-vector product, \f$ y = \alpha A x + \beta y \f$.

    @param[in]
    uplo    Whether the upper or lower triangle of A is referenced.

    @param[in]
    n       Number of rows and columns of A. n >= 0.

    @param[in]
    alpha   Scalar \f$ \alpha \f$

    @param[in]
    dA      REAL array of dimension (ldda,n), ldda >= max(1,n).
            The n-by-n matrix A, on GPU device.

    @param[in]
    ldda    Leading dimension of dA.

    @param[in]
    dx      REAL array on GPU device.
            The m element vector x of dimension (1 + (m-1)*incx).

    @param[in]
    incx    Stride between consecutive elements of dx. incx != 0.

    @param[in]
    beta    Scalar \f$ \beta \f$

    @param[in,out]
    dy      REAL array on GPU device.
            The n element vector y of dimension (1 + (n-1)*incy).

    @param[in]
    incy    Stride between consecutive elements of dy. incy != 0.

    @ingroup magma_sblas2
*/
extern "C" void
magma_ssymv(
    magma_uplo_t uplo,
    magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dA, size_t dA_offset, magma_int_t ldda,
    magmaFloat_const_ptr dx, size_t dx_offset, magma_int_t incx,
    float beta,
    magmaFloat_ptr       dy, size_t dy_offset, magma_int_t incy,
    magma_queue_t queue )
{
    if ( n <= 0 )
        return;

#define DEBUG_SSYMV
#ifdef  DEBUG_SSYMV
    const magma_int_t ione = 1;
    float *A, *x, *y;
    A = (float*) malloc( ldda * n * sizeof(float) );
    x = (float*) malloc( n * sizeof(float) );
    y = (float*) malloc( n * sizeof(float) );
    assert( A != NULL );
    assert( x != NULL );
    assert( y != NULL );
    magma_sgetmatrix( n, n, dA, dA_offset, ldda, A, ldda, queue );
    magma_sgetvector( n, dx, dx_offset, incx, x, 1, queue );
    magma_sgetvector( n, dy, dy_offset, incy, y, 1, queue );
    blasf77_ssymv( lapack_uplo_const(uplo), &n, &alpha, A, &ldda, x, &ione, &beta, y, &ione );
    magma_ssetvector( n, y, 1, dy, dy_offset, incy, queue );
    free( A );
    free( x );
    free( y );
#else
    cl_int err = clblasSsymv(
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
/** Perform symmetric rank-1 update, \f$ A = \alpha x x^H + A \f$.

    @param[in]
    uplo    Whether the upper or lower triangle of A is referenced.

    @param[in]
    n       Number of rows and columns of A. n >= 0.

    @param[in]
    alpha   Scalar \f$ \alpha \f$

    @param[in]
    dx      REAL array on GPU device.
            The n element vector x of dimension (1 + (n-1)*incx).

    @param[in]
    incx    Stride between consecutive elements of dx. incx != 0.

    @param[in,out]
    dA      REAL array of dimension (ldda,n), ldda >= max(1,n).
            The n-by-n matrix A, on GPU device.

    @param[in]
    ldda    Leading dimension of dA.

    @ingroup magma_sblas2
*/
extern "C" void
magma_ssyr(
    magma_uplo_t uplo,
    magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dx, size_t dx_offset, magma_int_t incx,
    magmaFloat_ptr       dA, size_t dA_offset, magma_int_t ldda,
    magma_queue_t queue )
{
    cl_int err = clblasSsyr(
        clblasColumnMajor,
        clblas_uplo_const( uplo ),
        n,
        alpha, dx, dx_offset, incx,
               dA, dA_offset, ldda,
        1, &queue, 0, NULL, g_event );
    check_error( err );
}

// --------------------
/** Perform symmetric rank-2 update, \f$ A = \alpha x y^H + conj(\alpha) y x^H + A \f$.

    @param[in]
    uplo    Whether the upper or lower triangle of A is referenced.

    @param[in]
    n       Number of rows and columns of A. n >= 0.

    @param[in]
    alpha   Scalar \f$ \alpha \f$

    @param[in]
    dx      REAL array on GPU device.
            The n element vector x of dimension (1 + (n-1)*incx).

    @param[in]
    incx    Stride between consecutive elements of dx. incx != 0.

    @param[in]
    dy      REAL array on GPU device.
            The n element vector y of dimension (1 + (n-1)*incy).

    @param[in]
    incy    Stride between consecutive elements of dy. incy != 0.

    @param[in,out]
    dA      REAL array of dimension (ldda,n), ldda >= max(1,n).
            The n-by-n matrix A, on GPU device.

    @param[in]
    ldda    Leading dimension of dA.

    @ingroup magma_sblas2
*/
extern "C" void
magma_ssyr2(
    magma_uplo_t uplo,
    magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dx, size_t dx_offset, magma_int_t incx,
    magmaFloat_const_ptr dy, size_t dy_offset, magma_int_t incy,
    magmaFloat_ptr       dA, size_t dA_offset, magma_int_t ldda,
    magma_queue_t queue )
{
    cl_int err = clblasSsyr2(
        clblasColumnMajor,
        clblas_uplo_const( uplo ),
        n,
        alpha, dx, dx_offset, incx,
               dy, dy_offset, incy,
               dA, dA_offset, ldda,
        1, &queue, 0, NULL, g_event );
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
    dA      REAL array of dimension (ldda,n), ldda >= max(1,n).
            The n-by-n matrix A, on GPU device.

    @param[in]
    ldda    Leading dimension of dA.

    @param[in]
    dx      REAL array on GPU device.
            The n element vector x of dimension (1 + (n-1)*incx).

    @param[in]
    incx    Stride between consecutive elements of dx. incx != 0.

    @ingroup magma_sblas2
*/
extern "C" void
magma_strmv(
    magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t n,
    magmaFloat_const_ptr dA, size_t dA_offset, magma_int_t ldda,
    magmaFloat_ptr       dx, size_t dx_offset, magma_int_t incx,
    magma_queue_t queue )
{
    if ( n <= 0 )
        return;

    magmaFloat_ptr dwork;
    magma_smalloc( &dwork, (1 + (n-1)*abs(incx)) );
    
    cl_int err = clblasStrmv(
        clblasColumnMajor,
        clblas_uplo_const( uplo ),
        clblas_trans_const( trans ),
        clblas_diag_const( diag ),
        n,
        dA, dA_offset, ldda,
        dx, dx_offset, incx,
        dwork,
        1, &queue, 0, NULL, g_event );
    clFlush(queue);
    check_error( err );
    
    magma_free( dwork );
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
    dA      REAL array of dimension (ldda,n), ldda >= max(1,n).
            The n-by-n matrix A, on GPU device.

    @param[in]
    ldda    Leading dimension of dA.

    @param[in,out]
    dx      REAL array on GPU device.
            On entry, the n element RHS vector b of dimension (1 + (n-1)*incx).
            On exit, overwritten with the solution vector x.

    @param[in]
    incx    Stride between consecutive elements of dx. incx != 0.

    @ingroup magma_sblas2
*/
extern "C" void
magma_strsv(
    magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t n,
    magmaFloat_const_ptr dA, size_t dA_offset, magma_int_t ldda,
    magmaFloat_ptr       dx, size_t dx_offset, magma_int_t incx,
    magma_queue_t queue )
{
    if ( n <= 0 )
        return;

    cl_int err = clblasStrsv(
        clblasColumnMajor,
        clblas_uplo_const( uplo ),
        clblas_trans_const( trans ),
        clblas_diag_const( diag ),
        n,
        dA, dA_offset, ldda,
        dx, dx_offset, incx,
        1, &queue, 0, NULL, g_event );
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
    dA      REAL array on GPU device.
            If transA == MagmaNoTrans, the m-by-k matrix A of dimension (ldda,k), ldda >= max(1,m); \n
            otherwise,                 the k-by-m matrix A of dimension (ldda,m), ldda >= max(1,k).

    @param[in]
    ldda    Leading dimension of dA.

    @param[in]
    dB      REAL array on GPU device.
            If transB == MagmaNoTrans, the k-by-n matrix B of dimension (lddb,n), lddb >= max(1,k); \n
            otherwise,                 the n-by-k matrix B of dimension (lddb,k), lddb >= max(1,n).

    @param[in]
    lddb    Leading dimension of dB.

    @param[in]
    beta    Scalar \f$ \beta \f$

    @param[in,out]
    dC      REAL array on GPU device.
            The m-by-n matrix C of dimension (lddc,n), lddc >= max(1,m).

    @param[in]
    lddc    Leading dimension of dC.

    @ingroup magma_sblas3
*/
extern "C" void
magma_sgemm(
    magma_trans_t transA, magma_trans_t transB,
    magma_int_t m, magma_int_t n, magma_int_t k,
    float alpha,
    magmaFloat_const_ptr dA, size_t dA_offset, magma_int_t ldda,
    magmaFloat_const_ptr dB, size_t dB_offset, magma_int_t lddb,
    float beta,
    magmaFloat_ptr       dC, size_t dC_offset, magma_int_t lddc,
    magma_queue_t queue )
{
    if ( m <= 0 || n <= 0 || k <= 0 )
        return;

    cl_int err = clblasSgemm(
        clblasColumnMajor,
        clblas_trans_const( transA ),
        clblas_trans_const( transB ),
        m, n, k,
        alpha, dA, dA_offset, ldda,
               dB, dB_offset, lddb,
        beta,  dC, dC_offset, lddc,
        1, &queue, 0, NULL, g_event );
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
    dA      REAL array on GPU device.
            If side == MagmaLeft, the m-by-m symmetric matrix A of dimension (ldda,m), ldda >= max(1,m); \n
            otherwise,            the n-by-n symmetric matrix A of dimension (ldda,n), ldda >= max(1,n).

    @param[in]
    ldda    Leading dimension of dA.

    @param[in]
    dB      REAL array on GPU device.
            The m-by-n matrix B of dimension (lddb,n), lddb >= max(1,m).

    @param[in]
    lddb    Leading dimension of dB.

    @param[in]
    beta    Scalar \f$ \beta \f$

    @param[in,out]
    dC      REAL array on GPU device.
            The m-by-n matrix C of dimension (lddc,n), lddc >= max(1,m).

    @param[in]
    lddc    Leading dimension of dC.

    @ingroup magma_sblas3
*/
extern "C" void
magma_ssymm(
    magma_side_t side, magma_uplo_t uplo,
    magma_int_t m, magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dA, size_t dA_offset, magma_int_t ldda,
    magmaFloat_const_ptr dB, size_t dB_offset, magma_int_t lddb,
    float beta,
    magmaFloat_ptr       dC, size_t dC_offset, magma_int_t lddc,
    magma_queue_t queue )
{
    cl_int err = clblasSsymm(
        clblasColumnMajor,
        clblas_side_const( side ),
        clblas_uplo_const( uplo ),
        m, n,
        alpha, dA, dA_offset, ldda,
               dB, dB_offset, lddb,
        beta,  dC, dC_offset, lddc,
        1, &queue, 0, NULL, g_event );
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
    dA      REAL array on GPU device.
            If trans == MagmaNoTrans, the n-by-k matrix A of dimension (ldda,k), ldda >= max(1,n); \n
            otherwise,                the k-by-n matrix A of dimension (ldda,n), ldda >= max(1,k).

    @param[in]
    ldda    Leading dimension of dA.

    @param[in]
    beta    Scalar \f$ \beta \f$

    @param[in,out]
    dC      REAL array on GPU device.
            The n-by-n symmetric matrix C of dimension (lddc,n), lddc >= max(1,n).

    @param[in]
    lddc    Leading dimension of dC.

    @ingroup magma_sblas3
*/
extern "C" void
magma_ssyrk(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    float alpha,
    magmaFloat_const_ptr dA, size_t dA_offset, magma_int_t ldda,
    float beta,
    magmaFloat_ptr       dC, size_t dC_offset, magma_int_t lddc,
    magma_queue_t queue )
{
    if (n <= 0 || k <= 0)
        return;

    cl_int err = clblasSsyrk(
        clblasColumnMajor,
        clblas_uplo_const( uplo ),
        clblas_trans_const( trans ),
        n, k,
        alpha, dA, dA_offset, ldda,
        beta,  dC, dC_offset, lddc,
        1, &queue, 0, NULL, g_event );
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
    dA      REAL array on GPU device.
            If trans == MagmaNoTrans, the n-by-k matrix A of dimension (ldda,k), ldda >= max(1,n); \n
            otherwise,                the k-by-n matrix A of dimension (ldda,n), ldda >= max(1,k).

    @param[in]
    ldda    Leading dimension of dA.

    @param[in]
    dB      REAL array on GPU device.
            If trans == MagmaNoTrans, the n-by-k matrix B of dimension (lddb,k), lddb >= max(1,n); \n
            otherwise,                the k-by-n matrix B of dimension (lddb,n), lddb >= max(1,k).

    @param[in]
    lddb    Leading dimension of dB.

    @param[in]
    beta    Scalar \f$ \beta \f$

    @param[in,out]
    dC      REAL array on GPU device.
            The n-by-n symmetric matrix C of dimension (lddc,n), lddc >= max(1,n).

    @param[in]
    lddc    Leading dimension of dC.

    @ingroup magma_sblas3
*/
extern "C" void
magma_ssyr2k(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    float alpha,
    magmaFloat_const_ptr dA, size_t dA_offset, magma_int_t ldda,
    magmaFloat_const_ptr dB, size_t dB_offset, magma_int_t lddb,
    float beta,
    magmaFloat_ptr       dC, size_t dC_offset, magma_int_t lddc,
    magma_queue_t queue )
{
    cl_int err = clblasSsyr2k(
        clblasColumnMajor,
        clblas_uplo_const( uplo ),
        clblas_trans_const( trans ),
        n, k,
        alpha, dA, dA_offset, ldda,
               dB, dB_offset, lddb,
        beta,  dC, dC_offset, lddc,
        1, &queue, 0, NULL, g_event );
    check_error( err );
}

#ifdef COMPLEX
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
    dA      REAL array on GPU device.
            If side == MagmaLeft, the m-by-m symmetric matrix A of dimension (ldda,m), ldda >= max(1,m); \n
            otherwise,            the n-by-n symmetric matrix A of dimension (ldda,n), ldda >= max(1,n).

    @param[in]
    ldda    Leading dimension of dA.

    @param[in]
    dB      REAL array on GPU device.
            The m-by-n matrix B of dimension (lddb,n), lddb >= max(1,m).

    @param[in]
    lddb    Leading dimension of dB.

    @param[in]
    beta    Scalar \f$ \beta \f$

    @param[in,out]
    dC      REAL array on GPU device.
            The m-by-n matrix C of dimension (lddc,n), lddc >= max(1,m).

    @param[in]
    lddc    Leading dimension of dC.

    @ingroup magma_sblas3
*/
extern "C" void
magma_ssymm(
    magma_side_t side, magma_uplo_t uplo,
    magma_int_t m, magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dA, size_t dA_offset, magma_int_t ldda,
    magmaFloat_const_ptr dB, size_t dB_offset, magma_int_t lddb,
    float beta,
    magmaFloat_ptr       dC, size_t dC_offset, magma_int_t lddc,
    magma_queue_t queue )
{
    if ( m <= 0 || n <= 0)
        return;

    cl_int err = clblasSsymm(
        clblasColumnMajor,
        clblas_side_const( side ),
        clblas_uplo_const( uplo ),
        m, n,
        alpha, dA, dA_offset, ldda,
               dB, dB_offset, lddb,
        beta,  dC, dC_offset, lddc,
        1, &queue, 0, NULL, g_event );
    clFlush(queue);
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
    dA      REAL array on GPU device.
            If trans == MagmaNoTrans, the n-by-k matrix A of dimension (ldda,k), ldda >= max(1,n); \n
            otherwise,                the k-by-n matrix A of dimension (ldda,n), ldda >= max(1,k).

    @param[in]
    ldda    Leading dimension of dA.

    @param[in]
    beta    Scalar \f$ \beta \f$

    @param[in,out]
    dC      REAL array on GPU device.
            The n-by-n symmetric matrix C of dimension (lddc,n), lddc >= max(1,n).

    @param[in]
    lddc    Leading dimension of dC.

    @ingroup magma_sblas3
*/
extern "C" void
magma_ssyrk(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    float alpha,
    magmaFloat_const_ptr dA, size_t dA_offset, magma_int_t ldda,
    float beta,
    magmaFloat_ptr       dC, size_t dC_offset, magma_int_t lddc,
    magma_queue_t queue )
{
    if (n <= 0 || k <= 0)
        return;

    cl_int err = clblasSsyrk(
        clblasColumnMajor,
        clblas_uplo_const( uplo ),
        clblas_trans_const( trans ),
        n, k,
        alpha, dA, dA_offset, ldda,
        beta,  dC, dC_offset, lddc,
        1, &queue, 0, NULL, g_event );
    clFlush(queue);
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
    dA      REAL array on GPU device.
            If trans == MagmaNoTrans, the n-by-k matrix A of dimension (ldda,k), ldda >= max(1,n); \n
            otherwise,                the k-by-n matrix A of dimension (ldda,n), ldda >= max(1,k).

    @param[in]
    ldda    Leading dimension of dA.

    @param[in]
    dB      REAL array on GPU device.
            If trans == MagmaNoTrans, the n-by-k matrix B of dimension (lddb,k), lddb >= max(1,n); \n
            otherwise,                the k-by-n matrix B of dimension (lddb,n), lddb >= max(1,k).

    @param[in]
    lddb    Leading dimension of dB.

    @param[in]
    beta    Scalar \f$ \beta \f$

    @param[in,out]
    dC      REAL array on GPU device.
            The n-by-n symmetric matrix C of dimension (lddc,n), lddc >= max(1,n).

    @param[in]
    lddc    Leading dimension of dC.

    @ingroup magma_sblas3
*/
extern "C" void
magma_ssyr2k(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    float alpha,
    magmaFloat_const_ptr dA, size_t dA_offset, magma_int_t ldda,
    magmaFloat_const_ptr dB, size_t dB_offset, magma_int_t lddb,
    float beta,
    magmaFloat_ptr dC, size_t dC_offset, magma_int_t lddc,
    magma_queue_t queue )
{
    if (n <= 0 || k <= 0)
        return;

    cl_int err = clblasSsyr2k(
        clblasColumnMajor,
        clblas_uplo_const( uplo ),
        clblas_trans_const( trans ),
        n, k,
        alpha, dA, dA_offset, ldda,
        dB, dB_offset, lddb,
        beta, dC, dC_offset, lddc,
        1, &queue, 0, NULL, g_event );
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
    dA      REAL array on GPU device.
            If side == MagmaLeft, the n-by-n triangular matrix A of dimension (ldda,n), ldda >= max(1,n); \n
            otherwise,            the m-by-m triangular matrix A of dimension (ldda,m), ldda >= max(1,m).

    @param[in]
    ldda    Leading dimension of dA.

    @param[in]
    dB      REAL array on GPU device.
            The m-by-n matrix B of dimension (lddb,n), lddb >= max(1,m).

    @param[in]
    lddb    Leading dimension of dB.

    @ingroup magma_sblas3
*/
extern "C" void
magma_strmm(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dA, size_t dA_offset, magma_int_t ldda,
    magmaFloat_ptr       dB, size_t dB_offset, magma_int_t lddb,
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
printf( "strmm( %c, %c, %c, %c, %d, %d, %d, %d )\n",
        lapacke_side_const(side),
        lapacke_uplo_const(uplo),
        lapacke_trans_const(trans),
        lapacke_diag_const(diag),
        m, n, ldda, lddb );
    float *A, *B;
    int k = (side == MagmaLeft ? m : n);
    A = (float*) malloc( ldda * k * sizeof(float) );
    B = (float*) malloc( lddb * n * sizeof(float) );
    assert( A != NULL );
    assert( B != NULL );
    magma_sgetmatrix( k, k, dA, dA_offset, ldda, A, ldda, queue );
    magma_sgetmatrix( m, n, dB, dB_offset, lddb, B, lddb, queue );
    blasf77_strmm( lapack_side_const(side), lapack_uplo_const(uplo),
                   lapack_trans_const(trans), lapack_diag_const(diag),
                   &m, &n, &alpha, A, &ldda, B, &lddb );
    magma_ssetmatrix( m, n, B, lddb, dB, dB_offset, lddb, queue );
    free( A );
    free( B );
printf( "strmm done\n" );
}
else {
#endif
    cl_int err = clblasStrmm(
        clblasColumnMajor,
        clblas_side_const( side ),
        clblas_uplo_const( uplo ),
        clblas_trans_const( trans ),
        clblas_diag_const( diag ),
        m, n,
        alpha, dA, dA_offset, ldda,
               dB, dB_offset, lddb,
        1, &queue, 0, NULL, g_event );
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
    dA      REAL array on GPU device.
            If side == MagmaLeft, the m-by-m triangular matrix A of dimension (ldda,m), ldda >= max(1,m); \n
            otherwise,            the n-by-n triangular matrix A of dimension (ldda,n), ldda >= max(1,n).

    @param[in]
    ldda    Leading dimension of dA.

    @param[in,out]
    dB      REAL array on GPU device.
            On entry, m-by-n matrix B of dimension (lddb,n), lddb >= max(1,m).
            On exit, overwritten with the solution matrix X.

    @param[in]
    lddb    Leading dimension of dB.

    @ingroup magma_sblas3
*/
extern "C" void
magma_strsm(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dA, size_t dA_offset, magma_int_t ldda,
    magmaFloat_ptr       dB, size_t dB_offset, magma_int_t lddb,
    magma_queue_t queue )
{
    if (m <= 0 || n <= 0)
        return;

#define DEBUG_TRSM
#ifdef  DEBUG_TRSM
printf( "strsm( %c, %c, %c, %c, %d, %d, %d, %d )\n",
        lapacke_side_const(side),
        lapacke_uplo_const(uplo),
        lapacke_trans_const(trans),
        lapacke_diag_const(diag),
        m, n, ldda, lddb );
    float *A, *B;
    int k = (side == MagmaLeft ? m : n);
    A = (float*) malloc( ldda * k * sizeof(float) );
    B = (float*) malloc( lddb * n * sizeof(float) );
    assert( A != NULL );
    assert( B != NULL );
//printf( "get dA( %d, %d, %p, %lu, %d, %p, %d, %p )\n", k, k, dA, dA_offset, ldda, A, ldda, queue );
    magma_sgetmatrix( k, k, dA, dA_offset, ldda, A, ldda, queue );
//printf( "get dB( %d, %d, %p, %lu, %d, %p, %d, %p )\n", m, n, dB, dB_offset, lddb, B, lddb, queue );
    magma_sgetmatrix( m, n, dB, dB_offset, lddb, B, lddb, queue );
//printf( "blasf77_strsm\n" );
    blasf77_strsm( lapack_side_const(side), lapack_uplo_const(uplo),
                   lapack_trans_const(trans), lapack_diag_const(diag),
                   &m, &n, &alpha, A, &ldda, B, &lddb );
//printf( "set dB( %d, %d, %p, %d, %p, %lu, %d, %p )\n", m, n, B, lddb, dB, dB_offset, lddb, queue );
    magma_ssetmatrix( m, n, B, lddb, dB, dB_offset, lddb, queue );
    free( A );
    free( B );
printf( "strsm done\n" );
#else
    cl_int err = clblasStrsm(
        clblasColumnMajor,
        clblas_side_const( side ),
        clblas_uplo_const( uplo ),
        clblas_trans_const( trans ),
        clblas_diag_const( diag ),
        m, n,
        alpha, dA, dA_offset, ldda,
               dB, dB_offset, lddb,
        1, &queue, 0, NULL, g_event );
    clFlush(queue);
    check_error( err );
#endif
}

#endif // HAVE_clblas

#undef REAL