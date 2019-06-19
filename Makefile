# ---------------------------------------------------------------------------
# programs
#
# Users should make all changes in make.inc
# It should not be necesary to change anything in here.

include make.inc

# defaults if nothing else is given in make.inc
CC         ?= cc
CXX        ?= c++
FORT       ?= no_fortran

ARCH       ?= ar
ARCHFLAGS  ?= cr
RANLIB     ?= ranlib

# shared libraries require -fPIC
#FPIC       = -fPIC

# may want -std=c99 for CFLAGS, -std=c++11 for CXXFLAGS
CFLAGS     ?= -O3 $(FPIC) -DADD_ -Wall -MMD
CXXFLAGS   ?= $(CFLAGS)
FFLAGS     ?= -O3 $(FPIC) -DADD_ -Wall -Wno-unused-dummy-argument
F90FLAGS   ?= -O3 $(FPIC) -DADD_ -Wall -Wno-unused-dummy-argument
LDFLAGS    ?= -O3 $(FPIC)

INC        ?= -I$(clBLAS)/include

LIBDIR     ?= -L$(clBLAS)/lib
LIB        ?= -lclBLAS -lOpenCL -llapack -lblas

# Extension for object files: o for unix, obj for Windows?
o_ext      ?= o

prefix     ?= /usr/local/clmagma


# ---------------------------------------------------------------------------
# MAGMA-specific programs & flags

ifeq ($(blas_fix),1)
    # prepend -lblas_fix to LIB (it must come before LAPACK library/framework)
    LIB := -L./lib -lblas_fix $(LIB)
endif

LIBS       = $(LIBDIR) $(LIB)

# preprocessor flags. See below for MAGMA_INC
CPPFLAGS   = $(INC) $(MAGMA_INC)

CFLAGS    += -DHAVE_clBLAS
CXXFLAGS  += -DHAVE_clBLAS

# where testers look for MAGMA libraries
RPATH      = -Wl,-rpath,../lib
RPATH2     = -Wl,-rpath,../../lib

codegen    = python tools/codegen.py

clcompile  = lib/clcompile


# ---------------------------------------------------------------------------
# Define the pointer size for fortran compilation
PTRFILE = control/sizeptr.c
PTROBJ  = control/sizeptr.$(o_ext)
PTREXEC = control/sizeptr
PTRSIZE = $(shell $(PTREXEC))
PTROPT  = -Dmagma_devptr_t="integer(kind=$(PTRSIZE))"

$(PTREXEC): $(PTROBJ)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $<


# ---------------------------------------------------------------------------
# include sub-directories

# variables that multiple sub-directories add to.
# these MUST be := defined, not = defined, for $(cdir) to work.
hdr                  :=
libmagma_src         :=
testing_src          :=
clcompile_src        :=

subdirs := \
	blas_fix            \
	control             \
	include             \
	interface_opencl    \
	src                 \
	clmagmablas         \
	testing             \
	testing/lin         \

Makefiles := $(addsuffix /Makefile.src, $(subdirs))

include $(Makefiles)

-include Makefile.internal
-include Makefile.local
-include Makefile.gen


# ---------------------------------------------------------------------------
# objects

ifeq ($(FORT),no_fortran)
    liblapacktest_all := $(filter     %_no_fortran.cpp, $(liblapacktest_all))
else
    liblapacktest_all := $(filter-out %_no_fortran.cpp, $(liblapacktest_all))
endif

ifeq ($(FORT),no_fortran)
    libmagma_all := $(filter-out %.f %.f90 %.F90, $(libmagma_all))
    testing_all  := $(filter-out %.f %.f90 %.F90, $(testing_all))
endif

# extract OpenCL sources
clkernels_all := $(filter     %.cl, $(libmagma_all))
libmagma_all  := $(filter-out %.cl, $(libmagma_all))
clkernels_obj := $(addsuffix .co, $(basename $(clkernels_all)))

libmagma_obj       := $(addsuffix .$(o_ext), $(basename $(libmagma_all)))
libblas_fix_obj    := $(addsuffix .$(o_ext), $(basename $(libblas_fix_src)))
libtest_obj        := $(addsuffix .$(o_ext), $(basename $(libtest_all)))
liblapacktest_obj  := $(addsuffix .$(o_ext), $(basename $(liblapacktest_all)))
testing_obj        := $(addsuffix .$(o_ext), $(basename $(testing_all)))
clcompile_obj      := $(addsuffix .$(o_ext), $(basename $(clcompile_src)))

deps :=
deps += $(addsuffix .d, $(basename $(libmagma_all)))
deps += $(addsuffix .d, $(basename $(libblas_fix_src)))
deps += $(addsuffix .d, $(basename $(libtest_all)))
deps += $(addsuffix .d, $(basename $(lapacktest_all)))
deps += $(addsuffix .d, $(basename $(testing_all)))

# headers must exist before compiling objects, but we don't want to require
# re-compiling the whole library for every minor header change,
# so use order-only prerequisite (after "|").
$(libmagma_obj):       | $(header_all)
$(libtest_obj):        | $(header_all)
$(testing_obj):        | $(header_all)

# changes to testings.h require re-compiling, e.g., if magma_opts changes
$(testing_obj):        testing/testings.h

# this allows "make force=force" to force re-compiling
$(libmagma_obj):       $(force)
$(libblas_fix_obj):    $(force)
$(libtest_obj):        $(force)
$(liblapacktest_obj):  $(force)
$(testing_obj):        $(force)

force: ;


# ----- include paths
MAGMA_INC  = -I./include -I./control

$(libtest_obj):        MAGMA_INC += -I./testing
$(testing_obj):        MAGMA_INC += -I./testing
$(libmagma_obj):       MAGMA_INC += -I./interface_opencl


# ----- headers
# to test that headers are self-contained,
# pre-compile each into a header.h.gch file using "g++ ... -c header.h"
header_gch := $(addsuffix .gch, $(filter-out %.cuh, $(header_all)))

test_headers: $(header_gch)

%.h.gch: %.h
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c -o $@ $<


# ----- libraries
libmagma_a      := lib/libclmagma.a
libmagma_so     := lib/libclmagma.so
libblas_fix_a   := lib/libblas_fix.a
libtest_a       := testing/libtest.a
liblapacktest_a := testing/lin/liblapacktest.a
libclkernels_co := lib/libclmagma_kernels.co

# static libraries
libs_a := \
	$(libmagma_a)		\
	$(libtest_a)		\
	$(liblapacktest_a)	\
	$(libblas_fix_a)	\

# shared libraries
libs_so := \
	$(libmagma_so)		\

# add objects to libraries
$(libmagma_a):      $(libmagma_obj)
$(libmagma_so):     $(libmagma_obj)
$(libblas_fix_a):   $(libblas_fix_obj)
$(libtest_a):       $(libtest_obj)
$(liblapacktest_a): $(liblapacktest_obj)
$(libclkernels_co): $(clkernels_obj)

# ----- testers
testing_c_src := $(filter %.c %.cpp,       $(testing_all))
testing_f_src := $(filter %.f %.f90 %.F90, $(testing_all))
testers       := $(basename $(testing_c_src))
testers_f     := $(basename $(testing_f_src))

# depend on static libraries
# see below for libmagma, which is either static or shared
$(testers):        $(libtest_a) $(liblapacktest_a)
$(testers_f):      $(libtest_a) $(liblapacktest_a)

# ----- blas_fix
# if using blas_fix (e.g., on MacOS), libmagma requires libblas_fix
ifeq ($(blas_fix),1)
    $(libmagma_a):     | $(libblas_fix_a)
    $(libmagma_so):    | $(libblas_fix_a)
    $(testers):        | $(libblas_fix_a)
    $(testers_f):      | $(libblas_fix_a)
    $(clcompile):      | $(libblas_fix_a)
endif


# ---------------------------------------------------------------------------
# MacOS likes shared library's path to be set; see make.inc.macos

ifneq ($(INSTALL_NAME),)
    $(libmagma_so):  LDFLAGS += $(INSTALL_NAME)$(notdir $(libmagma_so))
endif


# ---------------------------------------------------------------------------
# targets

.PHONY: all lib static shared clean test

.DEFAULT_GOAL := all

all: dense

dense: lib test

# lib defined below in shared libraries, depending on fPIC

test: testing

testers_f: $(testers_f)

# cleangen is defined in Makefile.gen; cleanall also does cleanmake in Makefile.internal
cleanall: clean cleangen

# TODO: should this do all $(subdirs) clean?
clean: lib/clean testing/clean
	-rm -f $(deps)


# ---------------------------------------------------------------------------
# shared libraries

# check whether all FLAGS have -fPIC
have_fpic = $(and $(findstring -fPIC, $(CFLAGS)),   \
                  $(findstring -fPIC, $(CXXFLAGS)), \
                  $(findstring -fPIC, $(FFLAGS)),   \
                  $(findstring -fPIC, $(F90FLAGS)))

ifneq ($(have_fpic),)
    # --------------------
    # if all flags have -fPIC: compile shared & static
    lib: static shared
    
    libs := $(libmagma_a) $(libmagma_so) $(libclkernels_co)

    shared: $(libmagma_so) $(libclkernels_co)
    
    # as a shared library, changing libmagma.so does NOT require re-linking testers,
    # so use order-only prerequisite (after "|").
    $(testers):        | $(libmagma_a) $(libmagma_so)
    $(testers_f):      | $(libmagma_a) $(libmagma_so)
    
    $(libmagma_so):
	@echo "===== shared library $@"
	$(CXX) $(LDFLAGS) -shared -o $@ \
		$^ \
		$(LIBS)
	@echo
else
    # --------------------
    # else: some flags are missing -fPIC: compile static only
    lib: static
    
    libs := $(libmagma_a) $(libclkernels_co)

    shared:
	@echo "Error: 'make shared' requires CFLAGS, CXXFLAGS, FFLAGS, F90FLAGS to have -fPIC."
	@echo "This is now the default in most example make.inc.* files, except atlas."
	@echo "Please edit your make.inc file and uncomment FPIC."
	@echo "After updating make.inc, please 'make clean && make shared && make test'."
	@echo "To compile only a static library, use 'make static'."
    
    # as a static library, changing libmagma.a does require re-linking testers,
    # so use regular prerequisite.
    $(testers):        $(libmagma_a)
    $(testers_f):      $(libmagma_a)
    
    # make libmagma.so ==> make shared ==> prints warning
    $(libs_so): shared
endif

ifeq ($(blas_fix),1)
    libs += $(libblas_fix_a)
endif


# ---------------------------------------------------------------------------
# static libraries

static: $(libmagma_a) $(libclkernels_co)

$(libs_a):
	@echo "===== static library $@"
	$(ARCH) $(ARCHFLAGS) $@ $^
	$(RANLIB) $@
	@echo


# ---------------------------------------------------------------------------
# sub-directory targets

control_obj          := $(filter          control/%.o, $(libmagma_obj))
interface_opencl_obj := $(filter interface_opencl/%.o, $(libmagma_obj))
clmagmablas_obj      := $(filter      clmagmablas/%.o, $(libmagma_obj))
src_obj              := $(filter              src/%.o, $(libmagma_obj))


# ----------
# sub-directory builds
include:             $(header_all)

blas_fix:            $(libblas_fix_a)

control:             $(control_obj)

interface_opencl:    $(interface_opencl_obj)

clmagmablas:         $(clmagmablas_obj)

src:                 $(src_obj)

testing:             $(testers)

# ----------
# sub-directory clean
include/clean:
	-rm -f $(shdr) $(dhdr) $(chdr)

blas_fix/clean:
	-rm -f $(libblas_fix_a) $(libblas_fix_obj)

control/clean:
	-rm -f $(control_obj) include/*.mod control/*.mod

interface_opencl/clean:
	-rm -f $(interface_opencl_obj)

clmagmablas/clean:
	-rm -f $(clmagmablas_obj)

src/clean:
	-rm -f $(src_obj)

testing/clean: testing/lin/clean
	-rm -f $(testers) $(testers_f) $(testing_obj) \
		$(libtest_a) $(libtest_obj)

testing/lin/clean:
	-rm -f $(liblapacktest_a) $(liblapacktest_obj)

# hmm... what should lib/clean do? just the libraries, not objects?
lib/clean: blas_fix/clean
	-rm -f $(libs) $(libmagma_obj)


# ---------------------------------------------------------------------------
# rules

.DELETE_ON_ERROR:

.SUFFIXES:

%.$(o_ext): %.f
	$(FORT) $(FFLAGS) -c -o $@ $<

%.$(o_ext): %.f90
	$(FORT) $(F90FLAGS) $(CPPFLAGS) -c -o $@ $<
	-mv $(notdir $(basename $@)).mod include/

%.$(o_ext): %.F90 $(PTREXEC)
	$(FORT) $(F90FLAGS) $(CPPFLAGS) $(PTROPT) -c -o $@ $<
	-mv $(notdir $(basename $@)).mod include/

%.$(o_ext): %.c
	$(CC) $(CFLAGS) $(CPPFLAGS) -c -o $@ $<

%.$(o_ext): %.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c -o $@ $<

%.i: %.h
	$(CC) -E $(CFLAGS) $(CPPFLAGS) -c -o $@ $<

%.i: %.c
	$(CC) -E $(CFLAGS) $(CPPFLAGS) -c -o $@ $<

%.i: %.cpp
	$(CXX) -E $(CXXFLAGS) $(CPPFLAGS) -c -o $@ $<


# ---------------------------------------------------------------------------
# OpenCL kernels

$(clcompile): $(clcompile_obj)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBS)

# TODO: clcompile doesn't currently update a file if it isn't out-of-date, so touch it.
%.co: %.cl | $(clcompile)
	$(clcompile) -c $< && touch $@

$(libclkernels_co): $(clkernels_obj)
	$(clcompile) -a -o $@ $^

clmagmablas/kernel_files.cpp: $(clkernels_all)
	perl tools/kernel_files.pl -o $@ $^


# ---------------------------------------------------------------------------
# testers

# link testing_foo from testing_foo.o
$(testers): %: %.$(o_ext)
	$(CXX) $(LDFLAGS) $(RPATH) \
	-o $@ $< \
	-L./lib -lclmagma \
	-L./testing -ltest \
	-L./testing/lin -llapacktest \
	$(LIBS)

# link Fortran testing_foo from testing_foo.o
$(testers_f): %: %.$(o_ext) testing/fortran.o
	$(FORT) $(LDFLAGS) $(RPATH) \
	-o $@ $< testing/fortran.o \
	-L./testing -ltest \
	-L./testing/lin -llapacktest \
	-L./lib -lclmagma \
	$(LIBS)


# ---------------------------------------------------------------------------
# filter out MAGMA-specific options for pkg-config
INSTALL_FLAGS := $(filter-out \
	-DMAGMA_NOAFFINITY -DMAGMA_SETAFFINITY -DMAGMA_WITH_ACML -DMAGMA_WITH_MKL -DUSE_FLOCK \
	-DHAVE_clBLAS \
	-fno-strict-aliasing -fPIC -O0 -O1 -O2 -O3 -pedantic -std=c99 -stdc++98 -stdc++11 \
	-Wall -Wshadow -Wno-long-long, $(CFLAGS))

INSTALL_LDFLAGS := $(filter-out -fPIC -Wall, $(LDFLAGS))

install_dirs:
	mkdir -p $(prefix)
	mkdir -p $(prefix)/include
	mkdir -p $(prefix)/lib
	mkdir -p $(prefix)/lib/pkgconfig

install: lib install_dirs
	# MAGMA
	cp include/*.h              $(prefix)/include
	cp $(libs)                  $(prefix)/lib
	# pkgconfig
	cat lib/pkgconfig/clmagma.pc.in                 | \
	sed -e s:@INSTALL_PREFIX@:"$(prefix)":          | \
	sed -e s:@CFLAGS@:"$(INSTALL_FLAGS) $(INC)":    | \
	sed -e s:@LIBS@:"$(INSTALL_LDFLAGS) $(LIBS)":   | \
	sed -e s:@MAGMA_REQUIRED@::                       \
	    > $(prefix)/lib/pkgconfig/clmagma.pc


# ---------------------------------------------------------------------------
# files.txt is nearly all (active) files in SVN, excluding directories. Useful for rsync, etc.
# files-doxygen.txt is all (active) source files in SVN, used by Doxyfile-fast

# excludes non-active directories like obsolete.
# excludes directories by matching *.* files (\w\.\w) and some exceptions like Makefile.
files.txt: force
	hg st -m -a -c \
		| perl -pi -e 's/^. +//' | sort \
		| egrep -v '^\.$$|obsolete|deprecated|contrib\b|^exp' \
		| egrep '\w\.\w|Makefile|docs|run' \
		> files.txt
	egrep -v '(\.html|\.css|\.f|\.in|\.m|\.mtx|\.pl|\.png|\.sh|\.txt)$$|checkdiag|COPYRIGHT|docs|example|make\.|Makefile|quark|README|Release|results|testing_|testing/lin|testing/matgen|tools' files.txt \
		| perl -pe 'chomp; $$_ = sprintf("\t../%-57s\\\n", $$_);' \
		> files-doxygen.txt

# files.txt per sub-directory
subdir_files = $(addsuffix /files.txt,$(subdirs))

$(subdir_files): force
	hg st -m -a -c -X '$(dir $@)/*/*' $(dir $@) \
		| perl -pi -e 's/^. +//' | sort \
		| perl -pi -e 's%^.{13} +\S+ +\S+ +\S+ +$(dir $@)%%' | sort \
		| egrep -v '^\.$$|obsolete|deprecated|contrib\b|^exp' \
		| egrep '\w\.\w|Makefile|docs|run' \
		> $@


# ---------------------------------------------------------------------------
echo:
	@echo "====="
	@echo "hdr                $(hdr)\n"
	@echo "header_all         $(header_all)\n"
	@echo "header_gch         $(header_gch)\n"
	@echo "====="
	@echo "libmagma_src       $(libmagma_src)\n"
	@echo "libmagma_all       $(libmagma_all)\n"
	@echo "libmagma_obj       $(libmagma_obj)\n"
	@echo "libmagma_a         $(libmagma_a)"
	@echo "libmagma_so        $(libmagma_so)"
	@echo "====="             
	@echo "blas_fix           $(blas_fix)"
	@echo "libblas_fix_src    $(libblas_fix_src)"
	@echo "libblas_fix_a      $(libblas_fix_a)"
	@echo "====="             
	@echo "libtest_src        $(libtest_src)\n"
	@echo "libtest_all        $(libtest_all)\n"
	@echo "libtest_obj        $(libtest_obj)\n"
	@echo "libtest_a          $(libtest_a)\n"
	@echo "====="
	@echo "liblapacktest_src  $(liblapacktest_src)\n"
	@echo "liblapacktest_all  $(liblapacktest_all)\n"
	@echo "liblapacktest_obj  $(liblapacktest_obj)\n"
	@echo "liblapacktest_a    $(liblapacktest_a)\n"
	@echo "====="
	@echo "testing_src        $(testing_src)\n"
	@echo "testing_all        $(testing_all)\n"
	@echo "testing_obj        $(testing_obj)\n"
	@echo "testers            $(testers)\n"
	@echo "testers_f          $(testers_f)\n"
	@echo "====="
	@echo "dep     $(dep)"
	@echo "deps    $(deps)\n"
	@echo "====="
	@echo "libs    $(libs)"
	@echo "libs_a  $(libs_a)"
	@echo "libs_so $(libs_so)"
	@echo "====="
	@echo "LIBS    $(LIBS)"
	@echo "====="
	@echo "clkernels_src      $(clkernels_src)"
	@echo "clkernels_obj      $(clkernels_obj)"
	@echo "libclkernels_co    $(libclkernels_co)"
	@echo "clcompile          $(clcompile)"
	@echo "clcompile_src      $(clcompile_src)"
	@echo "clcompile_obj      $(clcompile_obj)"


# ---------------------------------------------------------------------------
cmake:
	$(codegen) --srcdir ../magma-trunk --dstdir build --cmake $(libmagma_src) > cmake.txt


# ---------------------------------------------------------------------------
cleandep:
	-rm -f $(deps)

ifeq ($(dep),1)
    -include $(deps)
endif
