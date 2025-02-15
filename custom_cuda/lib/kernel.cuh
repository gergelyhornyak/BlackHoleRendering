#pragma once
#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include <cmath>
#include <png.h>
#include <string>
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// define realNumber data type, to be flexible for changes
typedef float realNumber;

#define OUTPUT_PNG_NAME "rendered_image.png"

__host__ __device__ struct pix_RGB {
    unsigned char R;
    unsigned char G;
    unsigned char B;
};

__host__ __device__ struct pix_realNumber {
    realNumber R;
    realNumber G;
    realNumber B;
};

// custom vector3D structure to hold 3D vector coordinates
__host__ __device__ struct vector3D {
    realNumber x;
    realNumber y;
    realNumber z;
};

__host__ __device__ struct config {
    vector3D camPos;
    int2 block_size;
    int2 grid_size;
    int img_w;
    int img_h;
    int ray_w;
    int ray_h;
    int ray_march_size;
    int num_of_threads;
    int focal_length;
    int num_of_runs;
};

cudaError_t renderPixelwCuda(pix_realNumber* h_pixels_raw, config * cfgs);
__host__ __device__ pix_realNumber pixel_function(int width, int height, config * cfgs);
__host__ void write_png_image(pix_RGB* pixels_all, char* filename, config* cfgs);
__host__ void image_generate_smooth(pix_realNumber* pixels_in, pix_RGB* pixels_out, config* cfgs);