#!/usr/bin/perl
#
# Usage:      cuda2opencl.pl foo.cu
# Generates:  foo.h, foo.cl, foo.cpp
#
# Does a quick job convert CUDA C driver to OpenCL C driver.
# Manually editing to cleanup the code will probably be required.
#
# Replaces dim3   (grid|threads)( ... );
# with     size_t (grid|threads)[dim] = { ... }
#
# Replaces func<<< grid, threads, 0, queue >>>( args );
# with     kernel = get_kernel("func");
#          err |= clSetKernelArg( kernel, i++, sizeof(arg), &arg );  // repeated for each arg in args
#          err  = clEnqueueNDRangeKernel( queue, kernel, dim, NULL, grid, threads, 0, NULL, NULL );
#
# Adds:    cl_kernel kernel;
#          cl_int err;
#          int i;
# at top of function.
#
# @author Mark Gates

# todo use extract_quotelike to get parenthesis correct?

use strict;
use List::Util qw(max);
use Text::Balanced qw(extract_bracketed extract_quotelike);


# -------------------------
# given C code, returns list of expressions.
# This is fairly rudimentary, but should handle spliting function arguments:
#     @args = parse_c( "foo, bar, min(m,n), max(i,j,k)" );
# returns ("foo", "bar", "min(m,n)", "max(i,j,k)").

sub parse_c
{
    my( $code ) = @_;
    my $sub;
    my @expr = ();
    my $expr = '';
    while( $code ) {
        if ( $code =~ s@^(\w+)@@ ) {
            $expr .= $1;
        }
        elsif ( $code =~ s@^(\s+)@@ ) {
            $expr .= $1;
        }
        elsif ( $code =~ s@^(//.*)@@ ) {
            $expr .= $1;
        }
        elsif ( $code =~ s@^(/\*.*?\*/)@@s ) {
            $expr .= $1;
        }
        elsif ( $code =~ s@^(\+=|-=|\*=|/=|%=|==|<=|>|>=)@@ ) {
            $expr .= $1;
        }
        elsif ( $code =~ s@^(\+|\-|\*|/|%|=|<|>)@@ ) {
            $expr .= $1;
        }
        elsif ( $code =~ m@^([\(\{\[])@ ) {
            ($sub, $code) = extract_bracketed( $code, $1 );
            $expr .= $sub;
        }
        elsif ( $code =~ m@^["']@ ) {  # '" # comment to fix jedit's syntax parser
            ($sub, $code) = extract_quotelike( $code );
            $expr .= $sub;
        }
        elsif ( $code =~ s@^,@@ ) {
            push @expr, $expr;
            $expr = '';
        }
        elsif ( $code =~ s@# *(define|include|undef|if|ifdef|ifndef|endif)\b.*@@ ) {
            push @expr, $expr;
            $expr = '';
        }
        else {
            $code =~ s@^(.)@@;
            $expr .= $1;
            print "unknown: $1\n";
        }
    }
    push @expr, $expr;
    return @expr;
}

# -------------------------
undef $/;
while( <> ) {
	my @blocks_h   = ();
	my @blocks_cl  = ();
	my @blocks_cpp = ();
	my $comment    = "";
	my $q = 0;
	while( $_ ) {
		if    ( m|^(\s*(?://.*\n)+)| ) {
			$_ = $';
			#print ">>> C++ comment: $1\n";
			$comment .= $1;
		}
		elsif ( m|^(\s*/\*.+?\*/\n?)|s ) {
			$_ = $';
			#print ">>> C comment: $1\n";
			$comment .= $1;
		}
		elsif ( m/^(\s*#define +(REAL|COMPLEX|PRECISION_\w) *\n)/ ) {
			$_ = $';
			push @blocks_cl,  $comment . $1;
			$comment = "";
		}
		elsif ( m/^(\s*#define.*\n)/ ) {
			$_ = $';
			push @blocks_h,   $comment . $1;
			$comment = "";
		}
		elsif ( m/^(\s*typedef +struct *\{.*?\} *\w+ *;\n)/s ) {
			$_ = $';
			#print ">>> struct: $1\n";
			push @blocks_h,   $comment . $1;
			$comment = "";
		}
		elsif ( m/^(\s*(extern "C"\s+)?(static\s+)?(?:__device__|__global__).*?\n{.*?\n\}[^\n]*\n)/s ) {
			$_ = $';
			#print ">>> kernel: $1\n";
			my $func = $1;
			if ( $func =~ m/\b(\w+)\(.*?\)\n\{/s ) {
				print "kernel: $1\n";
			}
			else {
				print "kernel: unknown\n";
			}
			push @blocks_cl,  $comment . $func;
			$comment = "";
		}
		elsif ( m|^(\s*extern "C".*?\n\{.*?\n\}[^\n]*\n)|s ) {
			$_ = $';
			#print ">>> extern: $1\n";
			my $func = $1;
			if ( $func =~ m/\b(\w+)\(.*?\)\n\{/s ) {
				print "driver: $1\n";
			}
			else {
				print "driver: unknown\n";
			}
			if ( $func =~ m/\bmagma\w+_q\(/ ) {           # record that this function is _q interface
				$q = 1;
			}
			my $stream = ($func =~ m/\bmagma_stream\b/);  # this function uses default stream
			if ( not $q or ($q and not $stream)) {        # if queue version exists, skip non-queue version
				push @blocks_cpp, $comment . $func;
			}
			else {
				print "        skipping non-queue version\n";
			}
			$comment = "";
		}
		elsif ( m|^(\s*.*\n)| ) {
			$_ = $';
			#print ">>> other:  $1\n";
			push @blocks_h,   $comment . $1;
			push @blocks_cl,  $comment . $1;
			push @blocks_cpp, $comment . $1;
			$comment = "";
		}
		else {
			print "rest: $_\n";
		}
	}
	
	
	my( $file, $cu_name, $h_name, $H_NAME, $cl_name, $cpp_name );
	$cu_name = $ARGV;
	$cu_name =~ s|^.*/||;  # chop off leading directories
	
	$h_name = $cu_name;
	$h_name =~ s/\.cu/.h/;
	print "writing $h_name\n";
	open( $file, ">$h_name" ) or die( $! );
	$H_NAME = "magma_" . $h_name;
	$H_NAME =~ tr/a-z/A-Z/;
	$H_NAME =~ s/\.H/_H/;
	print $file
	      "#ifndef $H_NAME\n",
	      "#define $H_NAME\n\n";
	foreach my $block ( @blocks_h ) {
		#print ">>>> block $block\n";
		output_header( $file, $block, $cu_name, $h_name );
	}
	print $file
	      "\n#endif // $H_NAME\n";
	close( $file );
	
	
	$cl_name = $cu_name;
	$cl_name =~ s/\.cu/.cl/;
	print "writing $cl_name\n";
	open( $file, ">$cl_name" ) or die( $! );
	foreach my $block ( @blocks_cl ) {
		#print ">>>> block $block\n";
		output_cl( $file, $block, $cu_name, $h_name );
	}
	close( $file );
	
	
	$cpp_name = $cu_name;
	$cpp_name =~ s/\.cu/.cpp/;
	print "writing $cpp_name\n";
	open( $file, ">$cpp_name" ) or die( $! );
	foreach my $block ( @blocks_cpp ) {
		#print ">>>> block $block\n";
		output_cpp( $file, $block, $cu_name, $h_name );
	}
	close( $file );
	
	print "\n";
}


# -------------------------
sub output_header
{
	my( $file, $_, $cu_name, $h_name ) = @_;
	
	# include note about auto-generation
	s/^( +)(\@precisions.*)/$1$2\n\n$1auto-converted from $cu_name/m;
	
	s/#include "common_magma.h" *\n//;
	s/#include "magma_templates.h" *\n//;
	
	print $file $_;
}


# -------------------------
sub output_cl
{
	my( $file, $_, $cu_name, $header ) = @_;
	
	# include note about auto-generation
	s/^( +)(\@precisions.*)/$1$2\n\n$1auto-converted from $cu_name/m;
	
	# prefix pointers (float*, etc.) with __global
	s/\b((?:float|double|magmaFloatComplex|magmaDoubleComplex|int|magma_int_t) *\* *(\w+))/__global $1/g;
	
	# pointer pointers
	s/\b((?:float|double|magmaFloatComplex|magmaDoubleComplex|int|magma_int_t) *\*)( *\* *(\w+))/__global $1 __global $2/g;
	
	# const pointer pointers
	s/\b((?:float|double|magmaFloatComplex|magmaDoubleComplex|int|magma_int_t) *const *\*)( *const *\* *(\w+))/__global $1 __global $2/g;
	
	# fix const __global  -->  __global const
	s/const __global/__global const/g;
	
	if ( s/\bstatic\b//g ) { print "  removing static\n"; }
	
	# change int to magma_int_t, on both device on kernel functions
	if ( m/((?:__device__|__global__)[^(]+)\((.*?)\)(\s*\{)/s ) {
		my $head = $1;
		my $args = $2;
		my $func = $3;
		$args =~ s/\bint\b/magma_int_t/g;
		s/((?:__device__|__global__)[^(]+)\((.*?)\)(\s*\{)/$head($args)$func/s;
	}
	
	# add offsets to pointer arguments, on kernel functions only (not device functions)
	if ( m/(__global__[^(]+)\((.*?)\)(\s*\{)/s ) {
		my $head = $1;
		my $args = $2;
		my $func = $3;
		my @args = split( /,/, $args );
		my $offsets = 0;
		for my $arg ( @args ) {
			if ( $arg =~ s/(__global [^,]+\b(\w+))/$1, unsigned long $2_offset/ ) {
				$func .= "\n    $2 += $2_offset;";
				$offsets = 1;
			}
		}
		if ( $offsets ) {
			$func .= "\n";
		}
		$args = join( ",", @args );
		s/(__global__[^(]+)\((.*?)\)(\s*\{)/$head($args)$func/s;
	}
	
	# map CUDA to OpenCL syntax
	s/__device__\s+//g;
	s/__global__/__kernel/g;
	s/__shared__/__local/g;
	s/__syncthreads\(\);/barrier( CLK_LOCAL_MEM_FENCE );/g;
	
	s/threadIdx\.x/get_local_id(0)/g;
	s/threadIdx\.y/get_local_id(1)/g;
	s/threadIdx\.z/get_local_id(2)/g;
	
	s/blockIdx\.x/get_group_id(0)/g;
	s/blockIdx\.y/get_group_id(1)/g;
	s/blockIdx\.z/get_group_id(2)/g;
	
	s/blockDim\.x/get_local_size(0)/g;
	s/blockDim\.y/get_local_size(1)/g;
	s/blockDim\.z/get_local_size(2)/g;
	
	s/gridDim\.x/get_num_groups(0)/g;
	s/gridDim\.y/get_num_groups(1)/g;
	s/gridDim\.z/get_num_groups(2)/g;
	
	# fix includes. Include function header last, e.g., zlascl.h
	s/#include "common_magma.h"/#include "kernels_header.h"/;
	s/#include "magma_templates.h"/#include "reduce.h"/;
	s/(.*\n)(#include[^\n]+)/$1$2\n#include "$header"/s;
	
	s/cuConj/MAGMA_Z_CNJG/g;
	s/cuCabs/MAGMA_Z_ABS/g;
	
	print $file $_;
}


# -------------------------
my @g_offsets;

sub add_offset
{
	my( $arg, $name, $post ) = @_;
	push @g_offsets, $name;
	return "$arg, size_t ${name}_offset$post";
}

# -------------------------
sub format_dim
{
	my( $space, $name, $args ) = @_;
	my @args = parse_c( $args );
	my $ndim = scalar(@args);  # i.e., length
	my $ret = $space . "size_t $name\[ndim];";
	for my $i ( 0 .. $ndim-1 ) {
		$args[$i] =~ s/^ +//;
		$args[$i] =~ s/ +$//;
		$ret .= "\n" . $space . "$name\[$i] = $args[$i];"
	}
	if ( $name =~ m/(grid|block)/ ) {
		for my $i ( 0 .. $ndim-1 ) {
			$ret .= "\n" . $space . "$name\[$i] *= threads[$i];"
		}
	}
	return $ret;
}

# -------------------------
sub output_cpp
{
	my( $file, $_, $cu_name, $header ) = @_;
	
	# include note about auto-generation
	s/^( +)(\@precisions.*)/$1$2\n\n$1auto-converted from $cu_name/m;
	
	# fix includes. Include function header last, e.g., zlascl.h
	s/(#include "common_magma.h")/#include "clmagma_runtime.h"\n$1/;
	s/#include "magma_templates.h" *\n//;
	s/(.*\n)(#include[^\n]+)/$1$2\n#include "$header"/s;
	
	# remove _q
	s/(magma\w+)_q\(/$1(/g;
	
	# double* foo  -->  ptr foo (should be fixed in CUDA)
	my $err = 0;
	if ( s/\bdouble( *)\*( *\w)/magmaDouble_ptr$1$2/g                ) { print "  ERROR: fix double*  =>  magmaDouble_ptr in CUDA code\n"; $err=1; }
	if ( s/\bfloat( *)\*( *\w)/magmaFloat_ptr$1$2/g                  ) { print "  ERROR: fix float*   =>  magmaFloat_ptr in CUDA code\n"; $err=1; }
	if ( s/\b(magma(?:Float|Double)Complex)( *)\*( *\w)/$1_ptr$2$3/g ) { print "  ERROR: fix $1  =>  $1_ptr in CUDA code\n"; $err=1; }
	if ( $err ) { die(); }
	
	# ptr  -->  ptr, offset
	my $cnt = 0;
	@g_offsets = ();
	if ( $cnt = s/(magma(?:Float|Double|FloatComplex|DoubleComplex|Int)(?:_const_ptr|_ptr)( +)(\w+))(,| *\))/add_offset( $1, $3, $4 )/ge ) {
		print "  added $cnt offsets in function args:  (@g_offsets)\n";
	}
	
	# TODO merge this stuff into format_dim to reduce duplicate processing
	# find # dimensions of dim3 variables
	my @dim3_args = m/dim3 +\w+\(([^;]*)\) *;/g;
	my @dims = ( 0, 0, 0, 0 );
	my $ndim = 0;
	foreach my $dim3 ( @dim3_args ) {
		my @args = parse_c( $dim3 );
		$ndim = scalar(@args);  # i.e., length
		$dims[ $ndim ] = 1;
	}
	if ( $dims[0] + $dims[1] + $dims[2] + $dims[3] > 1 ) {
		print STDERR "WARNING: multiple dimensions detected:";
		foreach my $i ( 0, 1, 2, 3 ) {
			if ( $dims[$i] > 0 ) {
				print STDERR " $i";
			}
		}
		print STDERR "\n";
	}
	
	# put ndim=[123] before first dim3
	s/^( *)(dim3)/${1}const int ndim = $ndim;\n$1$2/m;
	
	# dim3 grid( x, y )
	#   -->
	# size_t grid[2];
	# grid[0] = x;
	# grid[1] = y;
	#
	# if variable is grid|blocks, also add:
	# grid[0] *= threads[0];
	# grid[1] *= threads[1];
	s/^( *)dim3 +(\w+)\(([^;]*)\) *;/format_dim($1,$2,$3)/egm;
	
	s/^\{/{\n    cl_kernel kernel;\n    cl_int err;\n    int arg;\n/mg;
	
	# function<<< grid, threads, 0, queue >>>( args )
	#   -->
	# kernel = get_kernel("function");
	# clSetKernelArg( kernel, i++, sizeof(arg), &arg );  # repeat for each arg
	# clEnqueueNDRangeKernel( queue, kernel, dim, NULL, grid, threads, 0, NULL, NULL );
	my $post = $_;
	while( m/\G(.*?\n)( +)(\w+)\s*<<<\s*(\w+),\s*(\w+),\s*0,\s*(\w+)\s*>>>\s*\(\s*(.*?)\s*\);/sg )
	{
		if ( $ndim == 0 ) {
			print STDERR "WARNING: no dimension detected\n";
		}
		
		$post = $';
		my($pre, $i, $func, $grid, $threads, $queue, $args) = ($1, $2, $3, $4, $5, $6, $7);
		
		my $cnt2 = 0;
		my @k_offsets = ();
		foreach my $name ( @g_offsets ) {
			if ( $args =~ s/\b$name\b/$name, ${name}_offset/ ) {
				$cnt2 += 1;
				push @k_offsets, $name;
			}
		}
		print "  added $cnt2 offsets in calling kernel: (@k_offsets)\n";
		if ( $cnt != $cnt2 ) {
			print "  WARNING: doesn't match previous number of offsets in function arguments -- manually check!\n";
		}
		
		print $file $pre;
		print $file
		      "${i}kernel = g_runtime.get_kernel( \"$func\" );\n",
		      "${i}if ( kernel != NULL ) {\n",
		      "${i}    err = 0;\n",
		      "${i}    arg = 0;\n";
		my @args = split( ', +', $args );
		my $len = max( map( length, @args ));
		   $len = max( $len, 8 );
		for my $arg ( @args ) {
			printf $file
			       "${i}    err |= clSetKernelArg( kernel, arg++, sizeof(%-*s), &%-*s );\n",
			       $len, $arg, $len, $arg;
		}
		print $file
		      "${i}    check_error( err );\n",
		      "\n",
		      "${i}    err = clEnqueueNDRangeKernel( $queue, kernel, ndim, NULL, $grid, $threads, 0, NULL, NULL );\n",
		      "${i}    check_error( err );\n",
		      "${i}}";  # close if ( kernel )
	}
	print $file $post;  # post match
}
