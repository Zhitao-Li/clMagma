/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date
 
       @author Mark Gates
*/

#include <stdlib.h>
#include <stdio.h>

#include "clmagma_runtime.h"
#include "magma.h"
#include "error.h"

#ifdef HAVE_clBLAS

// ========================================
// memory allocation
// Allocate size bytes on GPU, returning pointer in ptrPtr.
extern "C" magma_int_t
magma_malloc( magma_ptr* ptrPtr, size_t size )
{
    // malloc and free sometimes don't work for size=0, so allocate some minimal size
    if ( size == 0 )
        size = sizeof(magmaDoubleComplex);
    cl_int err;
    *ptrPtr = clCreateBuffer( g_runtime.get_context(), CL_MEM_READ_WRITE, size, NULL, &err );
    if ( err != CLBlastSuccess ) {
        return MAGMA_ERR_DEVICE_ALLOC;
    }
    return MAGMA_SUCCESS;
}

// --------------------
// Free GPU memory allocated by magma_malloc.
extern "C" magma_int_t
magma_free( magma_ptr ptr )
{
    cl_int err = clReleaseMemObject( ptr );
    if ( err != CLBlastSuccess ) {
        return MAGMA_ERR_INVALID_PTR;
    }
    return MAGMA_SUCCESS;
}

// --------------------
// Allocate size bytes on CPU, returning pointer in ptrPtr.
// The purpose of using this instead of malloc() is to properly align arrays
// for vector (SSE) instructions. The default implementation uses
// posix_memalign (on Linux, MacOS, etc.) or _aligned_malloc (on Windows)
// to align memory to a 32 byte boundary.
// Use magma_free_cpu() to free this memory.
extern "C" magma_int_t
magma_malloc_cpu( void** ptrPtr, size_t size )
{
    // malloc and free sometimes don't work for size=0, so allocate some minimal size
    if ( size == 0 )
        size = sizeof(magmaDoubleComplex);
#if 1
    #if defined( _WIN32 ) || defined( _WIN64 )
    *ptrPtr = _aligned_malloc( size, 32 );
    if ( *ptrPtr == NULL ) {
        return MAGMA_ERR_HOST_ALLOC;
    }
    #else
    int err = posix_memalign( ptrPtr, 32, size );
    if ( err != 0 ) {
        *ptrPtr = NULL;
        return MAGMA_ERR_HOST_ALLOC;
    }
    #endif
#else
    *ptrPtr = malloc( size );
    if ( *ptrPtr == NULL ) {
        return MAGMA_ERR_HOST_ALLOC;
    }
#endif
    return MAGMA_SUCCESS;
}

// --------------------
// Free CPU pinned memory previously allocated by magma_malloc_pinned.
// The default implementation uses free(), which works for both malloc and posix_memalign.
// For Windows, _aligned_free() is used.
extern "C" magma_int_t
magma_free_cpu( void* ptr )
{
#if defined( _WIN32 ) || defined( _WIN64 )
    _aligned_free( ptr );
#else
    free( ptr );
#endif
    return MAGMA_SUCCESS;
}

#endif // HAVE_clBLAS
