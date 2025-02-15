# BlackHoleRender

CompSci Masters Semester One coursework, extending an existing project to render a black hole.

# IMPORTANT

For coursework moderation, consider the two following branches as the final versions of the project:

- [eigen-omp branch](https://github.coventry.ac.uk/hornyakj/7003_BlackHoleRender/tree/eigen_omp)
- [custom-cuda-antialias branch](https://github.coventry.ac.uk/hornyakj/7003_BlackHoleRender/tree/custom_cuda_antialias)

## Briefing

This project covers an implementation of a parallel-distributed version of the existing Black Hole Rendering program.


Each branch hosts a flavour of the original source code:

- [eigen branch](https://github.coventry.ac.uk/hornyakj/7003_BlackHoleRender/tree/eigen) with the original Eigen library,
- [eigen-omp branch](https://github.coventry.ac.uk/hornyakj/7003_BlackHoleRender/tree/eigen_omp) utilising the Eigen library and OpenMP library to create parallel processes,
- [custom-branch](https://github.coventry.ac.uk/hornyakj/7003_BlackHoleRender/tree/custom) which is customised with using self-written matrix transformation algorithms replacing Eigen library),
- [custom-omp branch](https://github.coventry.ac.uk/hornyakj/7003_BlackHoleRender/tree/custom_omp) an OpenMP flavour built on the custom branch. The final version of the CPU-based versions as well.
- [custom-omp-overloaded branch](https://github.coventry.ac.uk/hornyakj/7003_BlackHoleRender/tree/custom_omp_overloaded) experimenting with operator overloading, based on the custom_omp branch,
- [custom-cuda branch](https://github.coventry.ac.uk/hornyakj/7003_BlackHoleRender/tree/custom_cuda) a Cuda flavour, built on the custom branch, which is designed for running on GPUs,
- [custom-cuda-dynamic branch](https://github.coventry.ac.uk/hornyakj/7003_BlackHoleRender/tree/custom_cuda_dynamic) which focuses on dynamic grid and block size allocation,
- [custom-cuda-dynamic-coalesced branch](https://github.coventry.ac.uk/hornyakj/7003_BlackHoleRender/tree/custom_cuda_dynamic_coalesced) where experiments are carried out to see if memory is coalesced would result in faster computing on GPUs,
- [custom-cuda-dynamic-overloaded_branch](https://github.coventry.ac.uk/hornyakj/7003_BlackHoleRender/tree/custom_cuda_dynamic_overloaded) utilises built-in float4 and char4 structures to use SIMD parallel processing.
- [custom-cuda-antialias branch](https://github.coventry.ac.uk/hornyakj/7003_BlackHoleRender/tree/custom_cuda_antialias) branch is the final version of the GPU-based versions.

> each branch name includes its parent branches, whose features it builds upon

> complete project report can be found [here](7003_Report.pdf).

![black hole render](rendered_image.png)

## Requirements

The project was developed on Linux and Windows machines. 
All **cuda** branches utilising GPU, Windows environment, and Visual Studio 2022.
All the other branches mainly support Linux, therefore g++, gcc 
Windows: Visual Studio 2022 and support for Cuda Runtime v12.

### Windows

The cuda flavours require a dedicated GPU, and are developed in Visual Studio 2022, latter doesnt limit its compiling capabilities (still can be run from command line)

#### Command line

```
C:\Users\hornyakj\source\repos\CudaExercises>"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\nvcc.exe" -gencode=arch=compute_52,code=\"sm_52,compute_52\" --use-local-env -ccbin "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Tools\MSVC\14.41.34120\bin\HostX64\x64" -x cu   -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\include"  -G   --keep-dir CudaExercises\x64\Debug  -maxrregcount=0   --machine 64 --compile -cudart static  -g  -DWIN32 -DWIN64 -D_DEBUG -D_CONSOLE -D_MBCS -Xcompiler "/EHsc /W3 /nologo /Od /FS /Zi /RTC1 /MDd " -Xcompiler "/FdCudaExercises\x64\Debug\vc143.pdb" -o C:\Users\hornyakj\source\repos\CudaExercises\CudaExercises\x64\Debug\kernel.cu.obj "C:\Users\hornyakj\source\repos\CudaExercises\kernel.cu"
```

```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\nvcc.exe
-gencode=arch=compute_52,code=\"sm_52,compute_52\"
--use-local-env
-ccbin "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Tools\MSVC\14.41.34120\bin\HostX64\x64"
-x cu
-I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\include"
-G
--keep-dir CudaExercises\x64\Debug
-maxrregcount=0
--machine 64
--compile
-cudart static
```

#### Profiling

For profiling, Nvidia Nsight Compute 2024.3.2.0 was used.

### Linux (Ubuntu >=22)

On Linux, **g++** is suitable for compiling the *omp* and *serial* codes.

`g++ -O3 Functions.cpp main.cpp -o Executable -lpng -fopenmp`

> You will need libpng library installed
> g++ version >=10.x


`lscpu | grep 'Thread\|CPU(s):' | head -n 2`

For the Cuda flavour, you will need **nvcc** to compile the executable. 

Download the NVIDIA CUDA Toolkit

`nvcc` and add the flags from the [Windows instructions](#command-line)

## Project tasks

### Scope No1: 

- Report and Code: 
  - optimise:
    - for multiple cores of CPUs, ✅
    - and for GPUs, ✅
  - distribute the program using .cuh and .h files 
- Profile the code:
  - See how much of RAM, Cache, Registers are used,
  - Use profiling applications like **Nvidia Nsight Compute 2024.3.2.0** and **Intel VTune Profiler 2024.3**. ✅
  
### Scope No2: 

- Contribute to the project:
  - enhance the program with custom features
  - For example: refactor Eigen library, use instead own functions and methods, ✅
  - Use parameters through the code. ▶

## Reference

The default program originites from Aula, under the module page.

Papers:

https://oseiskar.github.io/black-hole/docs/physics.html

## Contributors

J. Gergely *Gary* Hornyak
