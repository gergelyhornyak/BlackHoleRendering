#include <chrono>
#include "kernel.cuh"

int main(int argc, char* argv[]) {

    // configuration set, to control constants
    //   best use as (command line) inputs
    config cfgs;
    // for anti-alias, shrink the output image by size 4
    cfgs.img_w = 1024*4/4;
    cfgs.img_h = 1024*4/4;
    cfgs.ray_w = 1024*4;
    cfgs.ray_h = 1024*4;
    cfgs.ray_march_size = 2048;
    cfgs.num_of_threads = 16;
    cfgs.camPos.x = 0.0f;
    cfgs.camPos.y = 7.0f;
    cfgs.camPos.z = 115.0f;
    cfgs.focal_length = 6.5f;
    cfgs.block_size.x = 32;
    cfgs.block_size.y = 32;
    cfgs.grid_size.x = 32*4;
    cfgs.grid_size.y = 32*4;
    cfgs.num_of_runs = 5;

    char PNG_filename[] = OUTPUT_PNG_NAME; // output file

    // declare dynamic arrays for raw pixels and RGB pixels
    pix_realNumber* pixels_raw = new pix_realNumber[cfgs.ray_h * cfgs.ray_w];
    pix_RGB* pixels_clean = new pix_RGB[cfgs.img_h * cfgs.img_w];

    printf("For benchmarking, do %d runs.\n",cfgs.num_of_runs);

    //Start timing rendering
    auto start = std::chrono::high_resolution_clock::now();

    // multiple runs of rendering, to create accurate average time for benchmarking
    //   due to CPU instruction caching
    for (int run = 0; run < cfgs.num_of_runs; run++) {
        cudaError_t cudaStatus = renderPixelwCuda(pixels_raw, &cfgs);
        if (cudaStatus != cudaSuccess) { // check for errors
            fprintf(stderr, "renderPixelwCuda failed!");
            return 1;
        }
    }
    //Stop timing
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    printf("Time [ms]: %ld (average of %d runs) \n",duration.count() / 1000 / cfgs.num_of_runs, cfgs.num_of_runs);

    // generate image using anti-alias
    image_generate_smooth(pixels_raw, pixels_clean, &cfgs);
    // export image as png
    write_png_image(pixels_clean, PNG_filename, &cfgs);

    //free dynamicly allocated memory
    delete[] pixels_raw;
    delete[] pixels_clean;

    return 0;
}