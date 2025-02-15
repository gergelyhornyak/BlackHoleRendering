#define _CRT_SECURE_NO_DEPRECATE
#include <stdlib.h>
#include <stdio.h>
#include <cstring>
#include <fstream>
#include <iostream>
#include "Functions.h"
#include "./eigen-3.4.0/Eigen/Dense"
#include <chrono>
#include <omp.h>

//Render
int main(){

    char PNG_filename[]="Out.png";
    pix_realNumber *pixels_raw;
    pixels_raw = (pix_realNumber*)malloc(sizeof(pix_realNumber)*RAY_HEIGHT*RAY_WIDTH);
    pix_RGB *pixels_clean;
    pixels_clean = (pix_RGB*)malloc(sizeof(pix_RGB)*IMAGE_HEIGHT*IMAGE_WIDTH);

    int num_of_max_threads = 16;

    //for(int t=1;t<=num_of_max_threads;t++)
    //{

    //Start timing
    auto start = std::chrono::high_resolution_clock::now();

    int x,y,index = 0;

    int num_of_threads = 16;
    int num_of_runs = 4;
    for(int run; run < num_of_runs; run++){
            
        #pragma omp parallel num_threads(num_of_threads)
        {
            #pragma omp for private(x) schedule(dynamic)
            //Write to each pixel (Array Index)
            for(y=0;y<RAY_HEIGHT;++y)
            {
                for(x=0;x<RAY_WIDTH;++x)
                {
                    //Do Work
                    pixels_raw[x+y*RAY_WIDTH]=pixel_function(x,y,RAY_HEIGHT,RAY_WIDTH);
                }
            }
        }
    }

    //Stop timing
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    printf("Time [ms]: %ld (average of %d runs)\n",duration.count() / 1000 / num_of_runs,num_of_runs);
    //}


    //Convert raw (realNumbers) pixels to proper RGB
    image_generate(pixels_raw, pixels_clean);
    //Write to binary file
    write_png_image(pixels_clean, PNG_filename);

    //Free Memory
    free(pixels_raw);
    free(pixels_clean);
}
