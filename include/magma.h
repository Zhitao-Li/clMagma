/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date
*/

#ifndef MAGMA_H
#define MAGMA_H

#include <math.h> // for sqrt, sqrtf
#include <vector>

/* ------------------------------------------------------------
 * MAGMA BLAS Functions
 * --------------------------------------------------------- */
#include "magmablas.h"

/* ------------------------------------------------------------
 * MAGMA functions
 * --------------------------------------------------------- */
#include "magma_z.h"
#include "magma_c.h"
#include "magma_d.h"
#include "magma_s.h"
#include "magma_zc.h"
#include "magma_ds.h"
#include "auxiliary.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ////////////////////////////////////////////////////////////////////////////
   -- MAGMA Auxiliary functions to get the NB used
*/
magma_int_t magma_get_smlsize_divideconquer();

// ========================================
// initialization
magma_int_t
magma_init( void );

magma_int_t
magma_init_1(std::vector<cl_device_id> devices, cl_context context);

magma_int_t
magma_init_opencl( cl_platform_id platform, cl_context context, magma_int_t setup_clBlas );

magma_int_t
magma_finalize( void );

magma_int_t
magma_finalize_opencl( magma_int_t finalize_clBlas );

void magma_version( magma_int_t* major, magma_int_t* minor, magma_int_t* micro );


// ========================================
// memory allocation
magma_int_t
magma_malloc( magma_ptr *ptrPtr, size_t bytes );

magma_int_t
magma_malloc_cpu( void **ptrPtr, size_t bytes );

//magma_int_t
//magma_malloc_pinned( void **ptrPtr, size_t bytes );

magma_int_t
magma_free( magma_ptr ptr );

magma_int_t
magma_free_cpu( void *ptr );

//magma_int_t
//magma_free_pinned( void *ptr );


// type-safe convenience functions to avoid using (void**) cast and sizeof(...)
// here n is the number of elements (floats, doubles, etc.) not the number of bytes.
static inline magma_int_t magma_imalloc( magmaInt_ptr           *ptrPtr, size_t n ) { return magma_malloc( (magma_ptr*) ptrPtr, n*sizeof(magma_int_t)        ); }
static inline magma_int_t magma_index_malloc( magmaIndex_ptr    *ptrPtr, size_t n ) { return magma_malloc( (magma_ptr*) ptrPtr, n*sizeof(magma_index_t)      ); }
static inline magma_int_t magma_smalloc( magmaFloat_ptr         *ptrPtr, size_t n ) { return magma_malloc( (magma_ptr*) ptrPtr, n*sizeof(float)              ); }
static inline magma_int_t magma_dmalloc( magmaDouble_ptr        *ptrPtr, size_t n ) { return magma_malloc( (magma_ptr*) ptrPtr, n*sizeof(double)             ); }
static inline magma_int_t magma_cmalloc( magmaFloatComplex_ptr  *ptrPtr, size_t n ) { return magma_malloc( (magma_ptr*) ptrPtr, n*sizeof(magmaFloatComplex)  ); }
static inline magma_int_t magma_zmalloc( magmaDoubleComplex_ptr *ptrPtr, size_t n ) { return magma_malloc( (magma_ptr*) ptrPtr, n*sizeof(magmaDoubleComplex) ); }

static inline magma_int_t magma_imalloc_cpu( magma_int_t        **ptrPtr, size_t n ) { return magma_malloc_cpu( (void**) ptrPtr, n*sizeof(magma_int_t)        ); }
static inline magma_int_t magma_index_malloc_cpu( magma_index_t **ptrPtr, size_t n ) { return magma_malloc_cpu( (void**) ptrPtr, n*sizeof(magma_index_t)      ); }
static inline magma_int_t magma_smalloc_cpu( float              **ptrPtr, size_t n ) { return magma_malloc_cpu( (void**) ptrPtr, n*sizeof(float)              ); }
static inline magma_int_t magma_dmalloc_cpu( double             **ptrPtr, size_t n ) { return magma_malloc_cpu( (void**) ptrPtr, n*sizeof(double)             ); }
static inline magma_int_t magma_cmalloc_cpu( magmaFloatComplex  **ptrPtr, size_t n ) { return magma_malloc_cpu( (void**) ptrPtr, n*sizeof(magmaFloatComplex)  ); }
static inline magma_int_t magma_zmalloc_cpu( magmaDoubleComplex **ptrPtr, size_t n ) { return magma_malloc_cpu( (void**) ptrPtr, n*sizeof(magmaDoubleComplex) ); }

//static inline magma_int_t magma_imalloc_pinned( magma_int_t        **ptrPtr, size_t n ) { return magma_malloc_pinned( (void**) ptrPtr, n*sizeof(magma_int_t)        ); }
//static inline magma_int_t magma_index_malloc_pinned( magma_index_t **ptrPtr, size_t n ) { return magma_malloc_pinned( (void**) ptrPtr, n*sizeof(magma_index_t)      ); }
//static inline magma_int_t magma_smalloc_pinned( float              **ptrPtr, size_t n ) { return magma_malloc_pinned( (void**) ptrPtr, n*sizeof(float)              ); }
//static inline magma_int_t magma_dmalloc_pinned( double             **ptrPtr, size_t n ) { return magma_malloc_pinned( (void**) ptrPtr, n*sizeof(double)             ); }
//static inline magma_int_t magma_cmalloc_pinned( magmaFloatComplex  **ptrPtr, size_t n ) { return magma_malloc_pinned( (void**) ptrPtr, n*sizeof(magmaFloatComplex)  ); }
//static inline magma_int_t magma_zmalloc_pinned( magmaDoubleComplex **ptrPtr, size_t n ) { return magma_malloc_pinned( (void**) ptrPtr, n*sizeof(magmaDoubleComplex) ); }


// ========================================
// device support
magma_int_t
magma_getdevices(
    magma_device_t* devices,
    magma_int_t     size,
    magma_int_t*    numPtr );


// ========================================
// queue support
magma_int_t
magma_queue_meminfo(magma_queue_t queue );

magma_int_t
magma_queue_create( magma_device_t device, magma_queue_t* queuePtr );

magma_int_t
magma_queue_destroy( magma_queue_t  queue );

magma_int_t
magma_queue_sync( magma_queue_t queue );


// ========================================
// event support
magma_int_t
magma_setevent( magma_event_t* eventPtr );

magma_int_t
magma_event_create( magma_event_t* eventPtr );

magma_int_t
magma_event_destroy( magma_event_t event );

magma_int_t
magma_event_record( magma_event_t event, magma_queue_t queue );

magma_int_t
magma_event_query( magma_event_t event );

magma_int_t
magma_event_sync( magma_event_t event );


// ========================================
// error handler
void magma_xerbla( const char *name, magma_int_t info );

const char* magma_strerror( magma_int_t error );


// ========================================
/// For integers x >= 0, y > 0, returns ceil( x/y ).
/// For x == 0, this is 0.
static inline magma_int_t magma_ceildiv( magma_int_t x, magma_int_t y )
{
    return (x + y - 1)/y;
}

/// For integers x >= 0, y > 0, returns x rounded up to multiple of y.
/// For x == 0, this is 0.
/// This implementation does not assume y is a power of 2.
static inline magma_int_t magma_roundup( magma_int_t x, magma_int_t y )
{
    return magma_ceildiv( x, y ) * y;
}


// ========================================
// real and complex square root
// sqrt alone cannot be caught by the generation script because of tsqrt
static inline float  magma_ssqrt( float  x ) { return sqrtf( x ); }
static inline double magma_dsqrt( double x ) { return sqrt( x ); }
magmaFloatComplex    magma_csqrt( magmaFloatComplex  x );
magmaDoubleComplex   magma_zsqrt( magmaDoubleComplex x );


#ifdef __cplusplus
}
#endif

#endif /* MAGMA_H */
