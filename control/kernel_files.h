#ifndef KERNELS_H
#define KERNELS_H

#ifdef __cplusplus
extern "C" {
#endif

struct kernel_files_t {
    const char* name;
    const char* file;
};

extern const struct kernel_files_t c_kernel_files[];

extern const int c_kernel_files_len;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif        //  #ifndef KERNELS_H
