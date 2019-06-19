#!/usr/bin/perl

use strict;
use warnings;
use Getopt::Std;

my $version;
my $major;
my $minor;
my $micro;

# default options
my $rc       = 0;  # release candidate
my $beta     = 0;

# in alphabetic order
my @files2delete = qw(
    BugsToFix.txt
    Makefile.gen
    Release-ToDo-1.1.txt
    Release-ToDo.txt
    docs
    include/Makefile
    interface_cuda
    interface_stub
    magmablas/obsolete
    quark
    sparse-iter
    src/obsolete
    testing/*.txt
    tools
);
# Using qw() avoids need for "quotes", but comments aren't recognized inside qw()
#src/magma_zf77.cpp
#src/magma_zf77pgi.cpp


sub myCmd
{
    my ($cmd) = @_ ;
    print "---------------------------------------------------------------\n";
    print $cmd."\n";
    print "---------------------------------------------------------------\n";
    my $err = system($cmd);
    if ($err != 0) {
        print "Error during execution of the following command:\n$cmd\n";
        exit;
    }
}

sub MakeRelease
{
    my $cmd;
    my $stage = "";

    if ( $rc > 0 ) {
        $version .= "-rc$rc";
        $stage = "rc$rc";
    }
    if ( $beta > 0 ) {
        $version .= "-beta$beta";
        $stage = "beta$beta";
    }

    my $RELEASE_PATH = $ENV{PWD}."/clmagma-$version";
    if ( -e $RELEASE_PATH ) {
        die( "RELEASE_PATH $RELEASE_PATH already exists.\nPlease delete it or use different version.\n" );
    }

    # Save current directory
    my $dir = `pwd`;
    chomp $dir;

    $cmd = "hg archive $RELEASE_PATH";
    myCmd($cmd);

    chdir $RELEASE_PATH;

    # Change version in magma.h
    myCmd("perl -pi -e 's/VERSION_MAJOR +[0-9]+/VERSION_MAJOR $major/' include/magma_types.h");
    myCmd("perl -pi -e 's/VERSION_MINOR +[0-9]+/VERSION_MINOR $minor/' include/magma_types.h");
    myCmd("perl -pi -e 's/VERSION_MICRO +[0-9]+/VERSION_MICRO $micro/' include/magma_types.h");
    myCmd("perl -pi -e 's/VERSION_STAGE +.+/VERSION_STAGE \"$stage\"/' include/magma_types.h");

    # Change the version and date in comments
    my($sec, $min, $hour, $mday, $mon, $year, $wday, $yday, $isdst) = localtime;
    my @months = (
        'January',   'February', 'March',    'April',
        'May',       'June',     'July',     'August',
        'September', 'October',  'November', 'December',
    );
    $year += 1900;
    my $date = "$months[$mon] $year";
    my $script = "s/MAGMA \\\(version [0-9.]+\\\)/MAGMA (version $version)/;";
    $script .= " s/\\\@date.*/\\\@date $date/;";
    myCmd("find . -type f -exec perl -pi -e '$script' {} \\;");
    
    # Change version in pkgconfig
    $script = "s/Version: [0-9.]+/Version: $version/;";
    myCmd("perl -pi -e '$script' lib/pkgconfig/clmagma.pc.in");
    
    # Precision Generation
    print "Generate the different precisions\n";
    myCmd("touch make.inc");
    myCmd("make -j generation");

    # Compile the documentation
    #print "Compile the documentation\n";
    #system("make -C ./docs");
    myCmd("rm -f make.inc");

    # Remove non-required files (e.g., Makefile.gen)
    foreach my $file (@files2delete) {
        myCmd("rm -rf $RELEASE_PATH/$file");
    }

    # Remove the lines relative to include directory in root Makefile
    myCmd("perl -ni -e 'print unless /cd include/' $RELEASE_PATH/Makefile");

    # Remove '.Makefile.gen files'
    myCmd("find $RELEASE_PATH -name .Makefile.gen -exec rm -f {} \\;");

    chdir $dir;

    # Save the InstallationGuide if we want to do a plasma-installer release
    #myCmd("cp $RELEASE_PATH/InstallationGuide README-installer");

    # Create tarball
    print "Create the tarball\n";
    my $DIRNAME  = `dirname $RELEASE_PATH`;
    my $BASENAME = `basename $RELEASE_PATH`;
    chomp $DIRNAME;
    chomp $BASENAME;
    myCmd("(cd $DIRNAME && tar cvzf ${BASENAME}.tar.gz $BASENAME)");
}

#sub MakeInstallerRelease {
#
#    my $version = "$major.$minor.$micro";
#    my $cmd;
#
#    $RELEASE_PATH = $ENV{ PWD}."/plasma-installer-$version";
#
#    # Sauvegarde du rep courant
#    my $dir = `pwd`;
#    chomp $dir;
#
#    $cmd = "hg archive $RELEASE_PATH";
#    myCmd($cmd);
#
#    # Save the InstallationGuide if we want to do a plasma-installer release
#    myCmd("cp README-installer $RELEASE_PATH/README");
#
#    #Create tarball
#    print "Create the tarball\n";
#    my $DIRNAME  = `dirname $RELEASE_PATH`;
#    my $BASENAME = `basename $RELEASE_PATH`;
#    chomp $DIRNAME;
#    chomp $BASENAME;
#    myCmd("(cd $DIRNAME && tar cvzf ${BASENAME}.tar.gz $BASENAME)");
#}

sub Usage
{
    print "MakeRelease.pl [options] major.minor.micro\n";
    print "   -h            Print this help\n";
    print "   -b beta       Beta version\n";
    print "   -c candidate  Release candidate number\n";
}

my %opts;
getopts("b:r:c:",\%opts);

if ( defined $opts{h}  ) {
    Usage();
    exit;
}
if ( defined $opts{c} ) {
    $rc = $opts{c};
}
if ( defined $opts{b} ) {
    $beta = $opts{b};
}
if ( ($#ARGV + 1) != 1 ) {
    Usage();
    exit;
}

$version = shift;
($major, $minor, $micro) = $version =~ m/^(\d+)\.(\d+)\.(\d+)$/;

MakeRelease();
#MakeInstallerRelease();
