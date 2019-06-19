#ifndef KERNELS_HEADER_H
#define KERNELS_HEADER_H

/*
 *   -- clMAGMA (version 0.1) --
 *      Univ. of Tennessee, Knoxville
 *      Univ. of California, Berkeley
 *      Univ. of Colorado, Denver
 *      @date
 *
 * @author Mark Gates
 */

// ========================================

#if defined(MAGMA_ILP64) || defined(MKL_ILP64)
//typedef int64_t magma_int_t;
typedef long long int magma_int_t;  // MKL uses long long int, not int64_t
#else
typedef int magma_int_t;
#endif

typedef float2 FloatComplex;
typedef double2 DoubleComplex;

typedef DoubleComplex magmaDoubleComplex;
typedef FloatComplex  magmaFloatComplex;

//static __inline FloatComplex
static inline FloatComplex
floatComplex(float real, float imag)
{
    FloatComplex z;
    z.x = real;
    z.y = imag;
    return z;
}

static inline DoubleComplex
doubleComplex(double real, double imag)
{
    DoubleComplex z;
    z.x = real;
    z.y = imag;
    return z;
}

// propagates inf and nan correctly
static inline double
magma_cabs(magmaDoubleComplex z)
{
    double x = fabs( z.x );
    double y = fabs( z.y );
    double big, small;
    if ( x > y ) {
        big   = x;
        small = y;
    }
    else {
        big   = y;
        small = x;
    }
    if ( big == 0 || isinf(big) ) {
        return big + small;  // add to propagate nan
    }
    small /= big;
    return big * sqrt( 1 + small*small );
}

static inline float
magma_cabsf(magmaFloatComplex z)
{
    // fabsf not found?
    float x = fabs( z.x );
    float y = fabs( z.y );
    float big, small;
    if ( x > y ) {
        big   = x;
        small = y;
    }
    else {
        big   = y;
        small = x;
    }
    if ( big == 0 || isinf(big) ) {
        return big + small;  // add to propagate nan
    }
    small /= big;
    // sqrtf not found?
    return big * sqrt( 1 + small*small );
}

/*
 * Multiply two complex numbers:
 *  a = (aReal + I*aImag)
 *  b = (bReal + I*bImag)
 *  a * b = (aReal + I*aImag) * (bReal + I*bImag)
 *        = aReal*bReal +I*aReal*bImag +I*aImag*bReal +I^2*aImag*bImag
 *        = (aReal*bReal - aImag*bImag) + I*(aReal*bImag + aImag*bReal)
*/
static inline FloatComplex cmul(FloatComplex a, FloatComplex b){
    return floatComplex( a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}

static inline DoubleComplex zmul(DoubleComplex a, DoubleComplex b){
    return doubleComplex( a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}

/*
 * Divide two complex numbers:
 *  aReal + I*aImag     (aReal + I*aImag) * (bReal - I*bImag)
 * ----------------- = ---------------------------------------
 *  bReal + I*bImag     (bReal + I*bImag) * (bReal - I*bImag)
 * 
 *        aReal*bReal - I*aReal*bImag + I*aImag*bReal - I^2*aImag*bImag
 *     = ---------------------------------------------------------------
 *            bReal^2 - I*bReal*bImag + I*bImag*bReal  -I^2*bImag^2
 * 
 *        aReal*bReal + aImag*bImag         aImag*bReal - Real*bImag 
 *     = ---------------------------- + I* --------------------------
 *            bReal^2 + bImag^2                bReal^2 + bImag^2
 * 
*/
static inline FloatComplex cdiv(FloatComplex a, FloatComplex b){
    return floatComplex((a.x*b.x + a.y*b.y)/(b.x*b.x + b.y*b.y), (a.y*b.x - a.x*b.y)/(b.x*b.x + b.y*b.y));
}

static inline DoubleComplex zdiv(DoubleComplex a, DoubleComplex b){
    return doubleComplex((a.x*b.x + a.y*b.y)/(b.x*b.x + b.y*b.y), (a.y*b.x - a.x*b.y)/(b.x*b.x + b.y*b.y));
}

#define MAGMA_Z_MAKE(r,i)     doubleComplex(r,i)
#define MAGMA_Z_REAL(a)       (a).x
#define MAGMA_Z_IMAG(a)       (a).y
#define MAGMA_Z_ADD(a, b)     MAGMA_Z_MAKE((a).x+(b).x, (a).y+(b).y)
#define MAGMA_Z_SUB(a, b)     MAGMA_Z_MAKE((a).x-(b).x, (a).y-(b).y)
#define MAGMA_Z_MUL(a, b)     zmul(a, b)
#define MAGMA_Z_DIV(a, b)     zdiv(a, b)
#define MAGMA_Z_CNJG(a)       MAGMA_Z_MAKE((a).x, -(a).y)
#define MAGMA_Z_ABS(a)        magma_cabs(a)

#define MAGMA_C_MAKE(r,i)     floatComplex(r,i)
#define MAGMA_C_REAL(a)       (a).x
#define MAGMA_C_IMAG(a)       (a).y
#define MAGMA_C_ADD(a, b)     MAGMA_C_MAKE((a).x+(b).x, (a).y+(b).y)
#define MAGMA_C_SUB(a, b)     MAGMA_C_MAKE((a).x-(b).x, (a).y-(b).y)
#define MAGMA_C_MUL(a, b)     cmul(a, b)
#define MAGMA_C_DIV(a, b)     cdiv(a, b)
#define MAGMA_C_CNJG(a)       MAGMA_C_MAKE((a).x, -(a).y)
#define MAGMA_C_ABS(a)        magma_cabsf(a)

#define MAGMA_Z_EQUAL(a,b)    (MAGMA_Z_REAL(a)==MAGMA_Z_REAL(b) && MAGMA_Z_IMAG(a)==MAGMA_Z_IMAG(b))

#define MAGMA_C_EQUAL(a,b)    (MAGMA_C_REAL(a)==MAGMA_C_REAL(b) && MAGMA_C_IMAG(a)==MAGMA_C_IMAG(b))

#define MAGMA_D_MAKE(r,i)     (r)
#define MAGMA_D_REAL(x)       (x)
#define MAGMA_D_IMAG(x)       (0.0f)
#define MAGMA_D_ADD(a, b)     ((a) + (b))
#define MAGMA_D_SUB(a, b)     ((a) - (b))
#define MAGMA_D_MUL(a, b)     ((a) * (b))
#define MAGMA_D_DIV(a, b)     ((a) / (b))
#define MAGMA_D_ABS(a)        ((a)>0?(a):-(a))
#define MAGMA_D_CNJG(a)       (a)
#define MAGMA_D_EQUAL(a,b)    ((a) == (b))

#define MAGMA_S_MAKE(r,i)     (r)
#define MAGMA_S_REAL(x)       (x)
#define MAGMA_S_IMAG(x)       (0.0)
#define MAGMA_S_ADD(a, b)     ((a) + (b))
#define MAGMA_S_SUB(a, b)     ((a) - (b))
#define MAGMA_S_MUL(a, b)     ((a) * (b))
#define MAGMA_S_DIV(a, b)     ((a) / (b))
#define MAGMA_S_ABS(a)        ((a)>0?(a):-(a))
#define MAGMA_S_CNJG(a)       (a)
#define MAGMA_S_EQUAL(a,b)    ((a) == (b))

#define MAGMA_Z_ZERO              MAGMA_Z_MAKE( 0.0, 0.0)
#define MAGMA_Z_ONE               MAGMA_Z_MAKE( 1.0, 0.0)
#define MAGMA_Z_HALF              MAGMA_Z_MAKE( 0.5, 0.0)
#define MAGMA_Z_NEG_ONE           MAGMA_Z_MAKE(-1.0, 0.0)
#define MAGMA_Z_NEG_HALF          MAGMA_Z_MAKE(-0.5, 0.0)
#define MAGMA_Z_NEGATE(a)         MAGMA_Z_MAKE(-(a).x, -(a).y)

#define MAGMA_C_ZERO              MAGMA_C_MAKE( 0.0, 0.0)
#define MAGMA_C_ONE               MAGMA_C_MAKE( 1.0, 0.0)
#define MAGMA_C_HALF              MAGMA_C_MAKE( 0.5, 0.0)
#define MAGMA_C_NEG_ONE           MAGMA_C_MAKE(-1.0, 0.0)
#define MAGMA_C_NEG_HALF          MAGMA_C_MAKE(-0.5, 0.0)
#define MAGMA_C_NEGATE(a)         MAGMA_C_MAKE(-(a).x, -(a).y)

#define MAGMA_D_ZERO              ( 0.0)
#define MAGMA_D_ONE               ( 1.0)
#define MAGMA_D_HALF              ( 0.5)
#define MAGMA_D_NEG_ONE           (-1.0)
#define MAGMA_D_NEG_HALF          (-0.5)
#define MAGMA_D_NEGATE(a)         (-(a))

#define MAGMA_S_ZERO              ( 0.0)
#define MAGMA_S_ONE               ( 1.0)
#define MAGMA_S_HALF              ( 0.5)
#define MAGMA_S_NEG_ONE           (-1.0)
#define MAGMA_S_NEG_HALF          (-0.5)
#define MAGMA_S_NEGATE(a)         (-(a))


#endif        //  #ifndef KERNELS_HEADER_H
