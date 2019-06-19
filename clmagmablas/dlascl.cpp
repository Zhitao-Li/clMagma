/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @generated from clmagmablas/zlascl.cpp, normal z -> d, Tue Jun 18 16:14:18 2019

       auto-converted from dlascl.cu


       @author Mark Gates
*/
#include "clmagma_runtime.h"
#include "common_magma.h"
#include "dlascl.h"


/**
    Purpose
    -------
    DLASCL multiplies the M by N real matrix A by the real scalar
    CTO/CFROM.  This is done without over/underflow as long as the final
    result CTO*A(I,J)/CFROM does not over/underflow. TYPE specifies that
    A may be full, upper triangular, lower triangular.

    Arguments
    ---------
    @param[in]
    type    magma_type_t
            TYPE indices the storage type of the input matrix A.
            = MagmaFull:   full matrix.
            = MagmaLower:  lower triangular matrix.
            = MagmaUpper:  upper triangular matrix.
            Other formats that LAPACK supports, MAGMA does not currently support.

    @param[in]
    kl      INTEGER
            Unused, for LAPACK compatability.

    @param[in]
    ku      KU is INTEGER
            Unused, for LAPACK compatability.

    @param[in]
    cfrom   DOUBLE PRECISION

    @param[in]
    cto     DOUBLE PRECISION
    \n
            The matrix A is multiplied by CTO/CFROM. A(I,J) is computed
            without over/underflow if the final result CTO*A(I,J)/CFROM
            can be represented without over/underflow.
            CFROM must be nonzero. CFROM and CTO must not be NAN.

    @param[in]
    m       INTEGER
            The number of rows of the matrix A.  M >= 0.

    @param[in]
    n       INTEGER
            The number of columns of the matrix A.  N >= 0.

    @param[in,out]
    dA      DOUBLE PRECISION array, dimension (LDDA,N)
            The matrix to be multiplied by CTO/CFROM.  See TYPE for the
            storage type.

    @param[in]
    ldda    INTEGER
            The leading dimension of the array A.  LDDA >= max(1,M).

    @param[out]
    info    INTEGER
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value.
    
    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_daux2
    ********************************************************************/
extern "C" void
magmablas_dlascl(
    magma_type_t type, magma_int_t kl, magma_int_t ku,
    double cfrom, double cto,
    magma_int_t m, magma_int_t n,
    magmaDouble_ptr dA, size_t dA_offset, magma_int_t ldda,
    magma_queue_t queue,
    magma_int_t *info )
{
    cl_kernel kernel;
    cl_int err;
    int arg;

    *info = 0;
    if ( type != MagmaLower && type != MagmaUpper && type != MagmaFull )
        *info = -1;
    else if ( cfrom == 0 || isnan(cfrom) )
        *info = -4;
    else if ( isnan(cto) )
        *info = -5;
    else if ( m < 0 )
        *info = -6;
    else if ( n < 0 )
        *info = -3;
    else if ( ldda < max(1,m) )
        *info = -7;
    
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return;  //info;
    }
    
    const int ndim = 1;
    size_t threads[ndim];
    threads[0] = NB;
    size_t grid[ndim];
    grid[0] = magma_ceildiv( m, NB );
    grid[0] *= threads[0];
    
    double smlnum, bignum, cfromc, ctoc, cto1, cfrom1, mul;
    magma_int_t done = false;
    
    // Uses over/underflow procedure from LAPACK dlascl
    // Get machine parameters
    smlnum = lapackf77_dlamch("s");
    bignum = 1 / smlnum;
    
    cfromc = cfrom;
    ctoc   = cto;
    int cnt = 0;
    while( ! done ) {
        cfrom1 = cfromc*smlnum;
        if ( cfrom1 == cfromc ) {
            // cfromc is an inf.  Multiply by a correctly signed zero for
            // finite ctoc, or a nan if ctoc is infinite.
            mul  = ctoc / cfromc;
            done = true;
            cto1 = ctoc;
        }
        else {
            cto1 = ctoc / bignum;
            if ( cto1 == ctoc ) {
                // ctoc is either 0 or an inf.  In both cases, ctoc itself
                // serves as the correct multiplication factor.
                mul  = ctoc;
                done = true;
                cfromc = 1;
            }
            else if ( fabs(cfrom1) > fabs(ctoc) && ctoc != 0 ) {
                mul  = smlnum;
                done = false;
                cfromc = cfrom1;
            }
            else if ( fabs(cto1) > fabs(cfromc) ) {
                mul  = bignum;
                done = false;
                ctoc = cto1;
            }
            else {
                mul  = ctoc / cfromc;
                done = true;
            }
        }
        
        if (type == MagmaLower) {
            kernel = g_runtime.get_kernel( "dlascl_lower" );
            if ( kernel != NULL ) {
                err = 0;
                arg = 0;
                err |= clSetKernelArg( kernel, arg++, sizeof(m        ), &m         );
                err |= clSetKernelArg( kernel, arg++, sizeof(n        ), &n         );
                err |= clSetKernelArg( kernel, arg++, sizeof(mul      ), &mul       );
                err |= clSetKernelArg( kernel, arg++, sizeof(dA       ), &dA        );
                err |= clSetKernelArg( kernel, arg++, sizeof(dA_offset), &dA_offset );
                err |= clSetKernelArg( kernel, arg++, sizeof(ldda     ), &ldda      );
                check_error( err );

                err = clEnqueueNDRangeKernel( queue, kernel, ndim, NULL, grid, threads, 0, NULL, NULL );
                check_error( err );
            }
        }
        else if (type == MagmaUpper) {
            kernel = g_runtime.get_kernel( "dlascl_upper" );
            if ( kernel != NULL ) {
                err = 0;
                arg = 0;
                err |= clSetKernelArg( kernel, arg++, sizeof(m        ), &m         );
                err |= clSetKernelArg( kernel, arg++, sizeof(n        ), &n         );
                err |= clSetKernelArg( kernel, arg++, sizeof(mul      ), &mul       );
                err |= clSetKernelArg( kernel, arg++, sizeof(dA       ), &dA        );
                err |= clSetKernelArg( kernel, arg++, sizeof(dA_offset), &dA_offset );
                err |= clSetKernelArg( kernel, arg++, sizeof(ldda     ), &ldda      );
                check_error( err );

                err = clEnqueueNDRangeKernel( queue, kernel, ndim, NULL, grid, threads, 0, NULL, NULL );
                check_error( err );
            }
        }
        else if (type == MagmaFull) {
            kernel = g_runtime.get_kernel( "dlascl_full" );
            if ( kernel != NULL ) {
                err = 0;
                arg = 0;
                err |= clSetKernelArg( kernel, arg++, sizeof(m        ), &m         );
                err |= clSetKernelArg( kernel, arg++, sizeof(n        ), &n         );
                err |= clSetKernelArg( kernel, arg++, sizeof(mul      ), &mul       );
                err |= clSetKernelArg( kernel, arg++, sizeof(dA       ), &dA        );
                err |= clSetKernelArg( kernel, arg++, sizeof(dA_offset), &dA_offset );
                err |= clSetKernelArg( kernel, arg++, sizeof(ldda     ), &ldda      );
                check_error( err );

                err = clEnqueueNDRangeKernel( queue, kernel, ndim, NULL, grid, threads, 0, NULL, NULL );
                check_error( err );
            }
        }
     
        cnt += 1;
    }
}
