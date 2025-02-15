#include "kernel.cuh"

__host__ __device__ void vector_normalize(pix_realNumber* vector)
{
    realNumber length = std::sqrt(vector->R * vector->R +
        vector->G * vector->G +
        vector->B * vector->B);

    if (length == 0) {
        vector->R = 0;
        vector->G = 0;
        vector->B = 0;
    }
    else {
        vector->R /= length;
        vector->G /= length;
        vector->B /= length;
    }
}

__host__ __device__ void vectors_addition(pix_realNumber* v1, pix_realNumber* v2, pix_realNumber* v3)
{
    // since pix_realNumber addition operator is overload, this function is obsolete.
    v3->R = v3->G = v3->B = 0.0f;
    v3->R = v1->R + v2->R;
    v3->G = v1->G + v2->G;
    v3->B = v1->B + v2->B;
}

__host__ __device__ void vectors_subtract(pix_realNumber* v1, pix_realNumber* v2, pix_realNumber* v3)
{
    v3->R = v3->G = v3->B = 0.0f;
    v3->R = v1->R - v2->R;
    v3->G = v1->G - v2->G;
    v3->B = v1->B - v2->B;
}

__host__ __device__ void vector_multiplication(pix_realNumber* v1, realNumber m)
{
    v1->R *= m;
    v1->G *= m;
    v1->B *= m;
}

__host__ __device__ pix_realNumber pixel_function(int width, int height, config * cfgs)
{
    pix_realNumber pixel;

    int maxWidth = cfgs->ray_w;
    int maxHeight = cfgs->ray_h;

    //Define UV Coordinates
    realNumber u = (realNumber(width) / realNumber(maxWidth)) - 0.5f;
    realNumber v = ((realNumber(maxHeight) - realNumber(height)) / realNumber(maxHeight)) - 0.5f;

    //Define ray
    realNumber FocalLength = cfgs->focal_length;
    pix_realNumber fragmentCoords;
    fragmentCoords.R = u * 0.01f;
    fragmentCoords.G = v * 0.01f;
    fragmentCoords.B = 0.0f;
    pix_realNumber camPos;
    camPos.R = cfgs->camPos.x;
    camPos.G = cfgs->camPos.y;
    camPos.B = cfgs->camPos.z;
    pix_realNumber rayOrig;
    rayOrig.R = 0.0f;
    rayOrig.G = 0.0f;
    rayOrig.B = (FocalLength / 1000.0f);
    pix_realNumber rayDir;
    rayDir.R = 0.0f;
    rayDir.G = 0.0f;
    rayDir.B = 0.0f;

    pix_realNumber rayOrig_copy;
    rayOrig_copy.R = rayOrig.R;
    rayOrig_copy.G = rayOrig.G;
    rayOrig_copy.B = rayOrig.B;
    vectors_addition(&rayOrig_copy, &camPos, &rayOrig);
    pix_realNumber temp_vect;
    vectors_addition(&fragmentCoords, &camPos, &temp_vect);
    vectors_subtract(&temp_vect, &rayOrig, &rayDir);
    vector_normalize(&rayDir);

    //Define Initials
    realNumber dx = 0.5f;
    realNumber r = 0.0f;
    realNumber G = 1.0f;
    realNumber M = 1.0f;
    pix_realNumber Temp;
    Temp.R = 0.0f;
    Temp.G = 0.0f;
    Temp.B = 0.0f;
    pix_realNumber dir;
    dir.R = 0.0f;
    dir.G = 0.0f;
    dir.B = 0.0f;

    realNumber mag = 0.0f;

    // for RGB values, changed to 255
    pixel.R = 1.0f / 255 * 10;
    pixel.G = 1.0f / 255 * 13;
    pixel.B = 1.0f / 255 * 22;

    pix_realNumber rayDir_copy;

    // dynamic memory allocation inside for loop can cause memory leak
    //   therefore using variables instead of pointers

    //Ray march
    int i = 0;
    for (i = 0; i < cfgs->ray_march_size; i++)
    {
        r = sqrt(rayOrig.R * rayOrig.R + rayOrig.G * rayOrig.G + rayOrig.B * rayOrig.B);
        dx = 0.1f;

        //Event Horizon Case
        if (r < 2.0f * G * M)
        {
            break; // pitch black
        }

        if (r > 120)
        {
            break; // space background
        }

        if (r > 10.0f && r < 70 && abs(rayOrig.G) < 0.2f)
        {
            // render yellowish disk
            pixel.R = 1.0f /255 * 255 / exp(pow((r - 40.0f) / 20.0f, 2));
            pixel.G = 1.0f /255 * 191  / exp(pow((r - 40.0f) / 20.0f, 2));
            pixel.B = 0.0f;
            break;
        }

        //Update Ray
        Temp.R = rayOrig.R;
        Temp.G = rayOrig.G;
        Temp.B = rayOrig.B;
        vector_normalize(&Temp);
        mag = (-2.0f * G * M) / (r * r);
        vector_multiplication(&Temp, (mag * 0.2f));
        vectors_addition(&rayDir, &Temp, &dir);
        vector_normalize(&dir);
        rayDir.R = dir.R;
        rayDir.G = dir.G;
        rayDir.B = dir.B;
        rayDir_copy.R = rayDir.R;
        rayDir_copy.G = rayDir.G;
        rayDir_copy.B = rayDir.B;
        vector_multiplication(&rayDir_copy, dx);
        rayOrig_copy.R = rayOrig.R;
        rayOrig_copy.G = rayOrig.G;
        rayOrig_copy.B = rayOrig.B;
        vectors_addition(&rayOrig_copy, &rayDir_copy, &rayOrig);
    }
    return pixel;
}

__device__ int getGlobalIdx_2D_2D() {
    // return the current thread ID
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
    return threadId;
}

__global__ void //__launch_bounds__(MAX_THREADS, MIN_BLOCKS) 
render_pixels_kernel_2D(pix_realNumber* pixels_raw, config * d_cfgs) {

    int index = getGlobalIdx_2D_2D();

    int x = index % d_cfgs->ray_h;
    int y = index / d_cfgs->ray_w;

    pixels_raw[index] = pixel_function(x, y, d_cfgs);
}

void kernel_pool(pix_realNumber* d_pixels_raw, config * d_cfgs, int2 block_size, int2 grid_size) {

    // allocate blocks and grids dimension dynamic, based on ray_w and ray_h
    dim3 sizeOfBlocks(block_size.x, block_size.y, 1);
    dim3 sizeOfGrids(grid_size.x, grid_size.y, 1);

    printf("\nDEBUG blockDim.x: %d | blockDim.y: %d | gridDim.x: %d | gridDim.y: %d \n\n", sizeOfBlocks.x, sizeOfBlocks.y, sizeOfGrids.x, sizeOfGrids.y);

    render_pixels_kernel_2D <<< sizeOfGrids, sizeOfBlocks >> > (d_pixels_raw, d_cfgs);
}

cudaError_t renderPixelwCuda(pix_realNumber* h_pixels_raw, config * h_cfgs) {

    // create pointers for array and struct
    pix_realNumber* d_pixels_raw;
    config* d_cfgs;

    cudaError_t cudaStatus;

    int numOfDevices;
    cudaGetDeviceCount(&numOfDevices);
    cudaStatus = cudaSetDevice(0);

    // allocate device memory for pixels array, and for configs struct
    cudaStatus = cudaMalloc((void**)&d_pixels_raw, h_cfgs->ray_h * h_cfgs->ray_w * sizeof(pix_realNumber));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&d_cfgs, sizeof(config));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // copy pixels array and configs struct from host to device memory
    cudaStatus = cudaMemcpy(d_pixels_raw, h_pixels_raw, h_cfgs->ray_w * h_cfgs->ray_h * sizeof(pix_realNumber), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(d_cfgs, h_cfgs, sizeof(config), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // start kernel pool function 
    kernel_pool(d_pixels_raw, d_cfgs, h_cfgs->block_size, h_cfgs->grid_size);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // copy pixels array values from device to host memory
    cudaStatus = cudaMemcpy(h_pixels_raw, d_pixels_raw, h_cfgs->ray_w * h_cfgs->ray_h * sizeof(pix_realNumber), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(d_pixels_raw);

    return cudaStatus;

}