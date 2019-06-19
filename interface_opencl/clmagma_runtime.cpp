#include <sys/stat.h>
#include <errno.h>

#include <map>
#include <string>
#include <vector>

#include "clmagma_runtime.h"

#include "kernel_files.h"


// ------------------------------------------------------------
// global runtime
clmagma_runtime g_runtime;


// ------------------------------------------------------------
/// Reads an entire file and returns its contents.
std::string read_file( const std::string& filename )
{
    std::string contents;
    
    FILE* file = fopen( filename.c_str(), "r" );
    if ( file == NULL ) {
        fprintf( stderr, "Can't open file '%s': %s (%d) at %s:%d\n",
                 filename.c_str(), strerror(errno), errno, __func__, __LINE__ );
        return contents;
    }
    
    // get file length, then rewind to beginning
    int  err  = fseek( file, 0, SEEK_END );
    long len  = ftell( file );
    int  err2 = fseek( file, 0, SEEK_SET );
    if ( err < 0 || len < 0 || err2 < 0 ) {
        fprintf( stderr, "Error seeking file '%s': %s (%d) at %s:%d\n",
                 filename.c_str(), strerror(errno), errno, __func__, __LINE__ );
        return contents;
    }
    
    // allocate & read data
    contents.resize( len );
    size_t len2 = fread( &contents[0], sizeof(char), len, file );
    if ( len != (long)len2 ) {
        fprintf( stderr, "Error reading file '%s': %s (%d) at %s:%d\n",
                 filename.c_str(), strerror(errno), errno, __func__, __LINE__ );
    }
    
    err = fclose( file );
    if ( err < 0 ) {
        fprintf( stderr, "Error closing file '%s': %s (%d) at %s:%d\n",
                 filename.c_str(), strerror(errno), errno, __func__, __LINE__ );
    }
    
    return contents;
}


// ------------------------------------------------------------
/// Returns true if a file exists.
bool path_exists( const std::string& file )
{
    struct stat s;
    int err = stat( file.c_str(), &s );
    return (err == 0);
}


// ------------------------------------------------------------
/// Returns file modification time if file exists,
/// else -1.
double path_modified( const std::string& file )
{
    struct stat s;
    int err = stat( file.c_str(), &s );
    if ( err == 0 )
        return s.st_mtime;
    else
        return -1;
}


// ------------------------------------------------------------
/// Joins directory and filename, returning "dir/file".
/// TODO need to deal with Windows?
std::string path_join(
    const std::string& dir,
    const std::string& file )
{
    return dir + '/' + file;
}


// ------------------------------------------------------------
/// Searches for file in paths, which is a colon-delimited list of directories,
/// similar to LD_LIBRARY_PATH.
/// Returns path to file, or empty string ("") if it isn't found.
std::string search_path(
    const std::string& file,
    const std::string& paths )
{
    // search current directory first
    if ( path_exists( file )) {
        return file;
    }
    
    size_t i1=0, i2=0;
    std::string path;
    while( i2 != std::string::npos ) {
        i2 = paths.find_first_of( ':', i1 );
        path = path_join( paths.substr(i1,i2-i1), file );
        if ( path_exists( path )) {
            return path;
        }
        i1 = i2+1;
    }
    return "";
}


// ------------------------------------------------------------
/// Find all occurances of string search in string data,
/// and replace with string replace. The string data is updated in-place.
void replace_all(
    const std::string& search,
    const std::string& replace,
    std::string& data )
{
    size_t i = 0;
    while( true ) {
        i = data.find( search, i );
        if ( i == std::string::npos )
            break;
        data.replace( i, search.size(), replace );
        i += replace.size();
    }
}


// ------------------------------------------------------------
/// Initialize clMagma runtime.
/// Queries for OpenCL platforms and devices, and creates an OpenCL context.
void clmagma_runtime::init( bool require_double )
{
    char device_name[1024];
    cl_int err;
    
    cl_uint num_platforms;
    cl_platform_id platform;
    err = clGetPlatformIDs( 1, &platform, &num_platforms );
    check_error( err );
    
    // TODO allocate m_devices. This next (commented out) line counts them.
    //err = clGetDeviceIDs( platform, CL_DEVICE_TYPE_GPU, 0, NULL, &m_num_devices );
    err = clGetDeviceIDs( platform, CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_ACCELERATOR, MAX_DEVICES, m_devices, &m_num_devices );
    check_error( err );
    
    // MAGMA requires double precision; skip devices that lack it.
    // Otherwise we get compile errors for some devices but not others,
    // which causes clGetProgramInfo( program, CL_PROGRAM_BINARY_SIZES, ... )
    // and such to fail (abort).
    if ( require_double ) {
        unsigned int good = 0;
        for( unsigned int dev=0; dev < m_num_devices; ++dev ) {
            cl_device_fp_config config;
            err = clGetDeviceInfo( m_devices[dev], CL_DEVICE_DOUBLE_FP_CONFIG, sizeof(config), &config, NULL );
            check_error( err );
            if ( config == 0 ) {
                clGetDeviceInfo( m_devices[dev], CL_DEVICE_NAME, sizeof(device_name), device_name,  NULL );
                //fprintf( stderr, "skippping device %s: doesn't support double precision\n", device_name );
            }
            else {
                // move good devices up
                if ( dev != good ) {
                    m_devices[good] = m_devices[dev];
                }
                ++good;
            }
        }
        m_num_devices = good;
    }
    
    //char driver[1024];
    //cl_ulong mem_size, alloc_size;
    //for( int dev=0; dev < m_num_devices; ++dev ) {
    //    clGetDeviceInfo( m_devices[dev], CL_DEVICE_NAME,               sizeof(device_name), device_name, NULL );
    //    clGetDeviceInfo( m_devices[dev], CL_DEVICE_GLOBAL_MEM_SIZE,    sizeof(mem_size),    &mem_size,   NULL );
    //    clGetDeviceInfo( m_devices[dev], CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(alloc_size),  &alloc_size, NULL );
    //    clGetDeviceInfo( m_devices[dev], CL_DRIVER_VERSION,            sizeof(driver),      driver,      NULL );
    //    printf( "Device %d: %-16s (memory %3.1f GiB, max allocation %3.1f GiB, driver %s)\n",
    //            dev, device_name, mem_size/(1024.*1024.*1024.), alloc_size/(1024.*1024.*1024.), driver );
    //}
    
    m_context = clCreateContext( NULL, m_num_devices, m_devices, NULL, NULL, &err );
    check_error( err );
    
    // create map from kernel name -> file name
    for( int i=0; i < c_kernel_files_len; ++i ) {
        m_kernel_files[ c_kernel_files[i].name ] = c_kernel_files[i].file;
    }
    
    // path to search for .co cached OpenCL objects and .cl source code
    const char* path = getenv( "CLMAGMA_PATH" );
    if ( path == NULL ) {
        path = getenv( "LD_LIBRARY_PATH" );
        if ( path == NULL ) {
            path = ".";
        }
    }
    m_path = path;
}


// ------------------------------------------------------------
/// Quit clMagma runtime.
/// Releases all kernels and the OpenCL context.
void clmagma_runtime::quit()
{
    cl_int err;
    std::map< std::string, cl_kernel >::iterator it, end;
    it  = m_kernels.begin();
    end = m_kernels.end();
    for( ; it != end; ++it ) {
        if ( (*it).second != NULL ) {
            err = clReleaseKernel( (*it).second );
            check_error( err );
        }
        m_kernels[ (*it).first ] = NULL;  // TODO how to delete entry from map?
    }
    m_kernels.clear();
    
    if ( m_context ) {
        err = clReleaseContext( m_context );
        check_error( err );
        m_context = NULL;
    }
}


// ------------------------------------------------------------
/// Looks up a kernel and compiles its file.
/// Stores kernels into m_kernels map, where get_kernel() can obtain them.
int clmagma_runtime::compile_kernel(
    const char* kernel )
{
    std::string file = m_kernel_files[ kernel ];
    if ( file == "" ) {
        fprintf( stderr, "Error: kernel '%s' not found in kernel_files map (check clmagmablas/kernels.cpp)\n", kernel );
        return MAGMA_ERR_NOT_FOUND;
    }
    
    std::string path = search_path( file, m_path );
    if ( path == "" ) {
        fprintf( stderr, "Error: file '%s' not found in $CLMAGMA_PATH or $LD_LIBRARY_PATH\n", file.c_str() );
        return MAGMA_ERR_NOT_FOUND;
    }
    
    return compile_file( path.c_str(), NULL );
}


// ------------------------------------------------------------
/// Compiles a single file containing OpenCL kernels.
/// infile  is the OpenCL file to read from, with extension .cl
/// outfile is the object file to write to,  with extension .co
/// If outfile is NULL, determines outfile name by replacing .cl with .co in infile.
/// Also stores kernels into m_kernels map, where get_kernel() can obtain them.
/// Prints compiler warnings & errors to stderr.
int clmagma_runtime::compile_file(
    const char* infile,
    const char* outfile )
{
    if ( m_context == NULL ) {
        fprintf( stderr, "Error in %s: runtime not initialized.\n", __func__ );
        return -1;
    }
    
    cl_program program = NULL;
    cl_int     err, build_err;
    char       data[ 32*1024 ];
    std::string src;
    
    //double start = get_wtime();
    //printf( "compiling %s\n", infile );
    
    // determine output file
    std::string outfile_str;
    if ( outfile != NULL ) {
        outfile_str = outfile;
    }
    else {
        outfile_str = infile;
        size_t found = outfile_str.find_last_of('.');
        if ( found == std::string::npos ) {
            outfile_str += ".co";
            fprintf( stderr, "Warning: no suffix found; saving to '%s'\n",
                     outfile_str.c_str() );
        }
        else {
            std::string ext = outfile_str.substr( found, outfile_str.size() );
            if ( ext != ".cl" ) {
                fprintf( stderr, "Warning: expected suffix '.cl'; got suffix '%s'\n",
                         ext.c_str() );
            }
            outfile_str.replace( found, outfile_str.size(), ".co" );
        }
    }
    
    // if output file exists and is newer than input file, just read it
    double t_src = path_modified( infile );
    double t_out = path_modified( outfile_str );
    if ( t_out > t_src ) {
        //printf( "using cached file %s\n", outfile_str.c_str() );
        const char* o = outfile_str.c_str();
        load_kernels( 1, &o );
        return 0;
    }

    // read input file
    src = read_file( infile );
    if ( src.size() == 0 ) {
        fprintf( stderr, "Error: empty file\n" );
        return -1;
    }
    
    // compile
    const char* src_str = src.c_str();
    program = clCreateProgramWithSource( m_context, 1, (const char**)&src_str, NULL, &err );
    check_error( err );
    
    std::string inc = infile;
    size_t i = inc.find_last_of( '/' );
    if ( i != std::string::npos ) {
        inc = "-I " + inc.substr( 0, i );
        //printf( "path %s, inc %s\n", infile, inc.c_str() );
    }
    else {
        inc = "-I .";
        //printf( "path %s, no / using inc %s\n", infile, inc.c_str() );
    }
    build_err = clBuildProgram( program, m_num_devices, m_devices, inc.c_str(), NULL, NULL );
    
    // print warnings & errors
    // oddly, even when err == 0, there can be errors for some devices (e.g., double not supported).
    for( unsigned int dev=0; dev < m_num_devices; ++dev ) {
        err = clGetProgramBuildInfo( program, m_devices[dev], CL_PROGRAM_BUILD_LOG, sizeof(data), data, NULL );
        check_error( err );
        // trim whitespace from log
        size_t len = strlen( data );
        while( len > 0 && isspace( data[len-1] )) {
            --len;
            data[len] = '\0';
        }
        if ( len > 0 ) {
            // \x1b[ starts ANSI color sequence, m ends sequence.
            // 31 is red text, 35 is magenta text, 1 is bold, 0 is reset.
            // see https://en.wikipedia.org/wiki/ANSI_escape_code
            std::string data_str = data;
            replace_all( "warning:", "\x1b[35;1m"  "warning:"  "\x1b[0m", data_str );
            replace_all( "error:",   "\x1b[31;1m"  "error:"    "\x1b[0m", data_str );
            fprintf( stderr, "\x1b[1m"
                     "%s: warnings and errors on device %d (clBuildProgram returned %d)"
                     "\x1b[0m"  "\n%s\n\n",
                     infile, dev, build_err, data_str.c_str() );
        }
    }
    
    // save compiled binary
    if ( build_err == 0 ) {
        std::vector< cl_program > programs( 1, program );
        load_kernels( programs );
        save_programs( programs, outfile_str.c_str() );
    }
    clReleaseProgram( program );
    program = NULL;
    
    //printf( "compile       time %.4f\n", get_wtime() - start );
    return build_err;
}


// ------------------------------------------------------------
/// Save a list of OpenCL programs (i.e., compiled binaries) to an object file.
void clmagma_runtime::save_programs(
    std::vector< cl_program >& programs,
    const char* filename )
{
    if ( m_context == NULL ) {
        fprintf( stderr, "Error in %s: runtime not initialized.\n", __func__ );
        return;
    }
    
    if ( programs.size() == 0 )
        return;
    
    //printf( "saving %s\n", filename );
    cl_int err;
    //double start = get_wtime();
    FILE* file = fopen( filename, "w" );
    if ( file == NULL ) {
        fprintf( stderr, "Can't open file '%s': %s (%d) at %s:%d\n",
                 filename, strerror(errno), errno, __func__, __LINE__ );
    }
    
    cl_uint num_programs = programs.size();
    std::vector< size_t >          binary_sizes( m_num_devices );
    std::vector< unsigned char* >  binaries    ( m_num_devices );
    
    // write header: # devices, # programs
    fwrite( &m_num_devices, sizeof(m_num_devices), 1, file );
    fwrite( &num_programs,  sizeof(num_programs),  1, file );
    
    for( unsigned int i=0; i < programs.size(); ++i ) {
        // verify program's number of devices
        cl_uint num_devices;
        clGetProgramInfo( programs[i], CL_PROGRAM_NUM_DEVICES, sizeof(num_devices), &num_devices, NULL );
        if ( num_devices != m_num_devices ) {
            fprintf( stderr, "Error: num_devices %u != m_num_devices %u\n",
                     num_devices, m_num_devices );
        }
        
        // write sizes
        err = clGetProgramInfo( programs[i], CL_PROGRAM_BINARY_SIZES, m_num_devices*sizeof(binary_sizes[0]), &binary_sizes[0], NULL );
        check_error( err );
        fwrite( &binary_sizes[0], sizeof(binary_sizes[0]), m_num_devices, file );
        
        // get, write, and free binaries
        for( unsigned int dev=0; dev < m_num_devices; dev++) {
            binaries[dev] = new unsigned char[binary_sizes[dev]];
        }
        err = clGetProgramInfo( programs[i], CL_PROGRAM_BINARIES, m_num_devices*sizeof(binaries[0]), &binaries[0], NULL );
        check_error( err );
        for( unsigned int dev=0; dev < m_num_devices; ++dev ) {
            fwrite( binaries[dev], sizeof(binaries[dev][0]), binary_sizes[dev], file );
        }
        for( unsigned int dev=0; dev < m_num_devices; ++dev ) {
            delete [] binaries[dev];
            binaries[dev] = NULL;
        }
    }
    
    err = fclose( file );
    if ( err != 0 ) {
        fprintf( stderr, "Error closing file\n" );
    }
    //printf( "save programs time %.4f\n", get_wtime() - start );
}


// ------------------------------------------------------------
/// Read a list of OpenCL programs (i.e., compiled binaries) from a list of
/// object files. Files are searched for in the
/// $CLMAGMA_PATH or $LD_LIBRARY_PATH from the user's environment.
/// Devices that were compiled for must match the currently available devices.
/// @see load_kernels
void clmagma_runtime::load_programs(
    int nfiles, const char* const* infiles,
    std::vector< cl_program >& programs )
{
    if ( m_context == NULL ) {
        fprintf( stderr, "Error in %s: runtime not initialized.\n", __func__ );
        return;
    }
    
    cl_program program = NULL;
    cl_int     err;
    char       data[ 32*1024 ];
    std::string paths_str, path_str;
    
    //double start = get_wtime();
    for( int i=0; i < nfiles; ++i ) {
        // look for file in path
        path_str = search_path( infiles[i], m_path );
        if ( path_str == "" ) {
            fprintf( stderr, "Error: file '%s' not found in $CLMAGMA_PATH or $LD_LIBRARY_PATH\n", infiles[i] );
            continue;
        }
        //printf( "reading '%s'\n", path_str.c_str() );
        
        // open file
        FILE* file = fopen( path_str.c_str(), "r" );
        if ( file == NULL ) {
            fprintf( stderr, "Can't open file '%s': %s (%d) at %s:%d\n",
                     path_str.c_str(), strerror(errno), errno, __func__, __LINE__ );
            continue;
        }
        
        // read # devices
        // TODO read some description of devices & compare to current devices
        size_t len;
        cl_uint num_devices;
        len = fread( &num_devices, sizeof(num_devices), 1, file );
        if ( len != 1 ) { fprintf( stderr, "Error reading num devices\n" ); }
        if ( num_devices != m_num_devices ) {
            fprintf( stderr, "Error: num_devices %u does not match current m_num_devices %u\n",
                     num_devices, m_num_devices );
        }
        
        // read # programs
        cl_uint num_programs;
        len = fread( &num_programs, sizeof(num_programs), 1, file );
        if ( len != 1 ) { fprintf( stderr, "Error reading num programs\n" ); }
        
        // for each program
        //     read sizes[ 0:ndevices ]
        //     for each dev
        //         read bin[dev], size[dev]
        std::vector< size_t >         binary_sizes( m_num_devices );
        std::vector< unsigned char* > binaries    ( m_num_devices );
        std::vector< cl_int >         statuses    ( m_num_devices );
        for( unsigned int p=0; p < num_programs; ++p ) {
            len = fread( &binary_sizes[0], sizeof(binary_sizes[0]), m_num_devices, file );
            if ( len != m_num_devices ) { fprintf( stderr, "Error reading sizes\n" ); }
            
            for( unsigned int dev=0; dev < m_num_devices; ++dev ) {
                binaries[dev] = new unsigned char[ binary_sizes[dev] ];
                len = fread( binaries[dev], sizeof(binaries[dev][0]), binary_sizes[dev], file );
                if ( len != binary_sizes[dev] ) { fprintf( stderr, "Error reading binaries\n" ); }
            }
            
            program = clCreateProgramWithBinary(
                m_context, m_num_devices, m_devices,
                &binary_sizes[0], (const unsigned char**) &binaries[0], &statuses[0], &err );
            check_error( err );
            
            err = clBuildProgram( program, 0, NULL, NULL, NULL, NULL );
            check_error( err );
            for( unsigned int dev=0; dev < m_num_devices; ++dev ) {
                err = clGetProgramBuildInfo( program, m_devices[dev], CL_PROGRAM_BUILD_LOG, sizeof(data), data, NULL );
                check_error( err );
                if ( strlen( data ) > 0 ) {
                    fprintf( stderr, "compile errors on device %d:\n%s\n----\n",
                             dev, data );
                }
            }
            
            for( unsigned int dev=0; dev < m_num_devices; ++dev ) {
                delete [] binaries[dev];
                binaries[dev] = NULL;
            }
            
            if ( err == 0 ) {
                programs.push_back( program );
            }
        }
    }
    //printf( "load programs time %.4f\n", get_wtime() - start );
}


// ------------------------------------------------------------
/// Combine several object files into one archived object file.
void clmagma_runtime::archive_files(
    int nfiles, const char* const* infiles,
    const char* outfile )
{
    if ( m_context == NULL ) {
        fprintf( stderr, "Error in %s: runtime not initialized.\n", __func__ );
        return;
    }
    
    //printf( "archiving to %s\n", outfile );
    std::vector< cl_program > programs;
    load_programs( nfiles, infiles, programs );
    save_programs( programs, outfile );
    for( unsigned int i=0; i < programs.size(); ++i ) {
        clReleaseProgram( programs[i] );
        programs[i] = NULL;
    }
}


// ------------------------------------------------------------
/// Reads OpenCL kernels from a list of object files.
/// Stores kernels into m_kernels map, where get_kernel() can obtain them.
/// Internally, loads OpenCL programs, then takes the kernel data from the
/// programs and creates kernels.
/// @see magma_init()
void clmagma_runtime::load_kernels(
    int nfiles, const char* const* infiles )
{
    if ( m_context == NULL ) {
        fprintf( stderr, "Error in %s: runtime not initialized.\n", __func__ );
        return;
    }
    
    //double start = get_wtime();
    std::vector< cl_program > programs;
    std::vector< cl_kernel  > kernels;
    load_programs( nfiles, infiles, programs );
    load_kernels( programs );
    
    for( unsigned int i=0; i < programs.size(); ++i ) {
        clReleaseProgram( programs[i] );
        programs[i] = NULL;
    }
    //printf( "load kernels  time %.4f\n", get_wtime() - start );
}


// ------------------------------------------------------------
/// Reads OpenCL kernels from a list of programs (i.e., compiled binaries).
/// Stores the kernels into m_kernels map, where get_kernel() can obtain them.
void clmagma_runtime::load_kernels(
    const std::vector< cl_program >& programs )
{
    if ( m_context == NULL ) {
        fprintf( stderr, "Error in %s: runtime not initialized.\n", __func__ );
        return;
    }
    
    cl_uint num_kernels;
    cl_int  err;
    char    data[ 32*1024 ];
    std::vector< cl_kernel > kernels;
    
    //double start = get_wtime();
    for( unsigned int i=0; i < programs.size(); ++i ) {
        // this implementation works with OpenCL 1.0 (previously required OpenCL 1.2)
        // query # kernels
        err = clCreateKernelsInProgram( programs[i], 0, NULL, &num_kernels );
        check_error( err );
        
        kernels.resize( num_kernels );
        err = clCreateKernelsInProgram( programs[i], num_kernels, &kernels[0], NULL );
        check_error( err );
            
        for( unsigned int j=0; j < num_kernels; ++j ) {
            err = clGetKernelInfo( kernels[j], CL_KERNEL_FUNCTION_NAME, sizeof(data), data, NULL );
            check_error( err );
            m_kernels[ data ] = kernels[j];
        }
    }
    //printf( "load kernels  time %.4f\n", get_wtime() - start );
}
