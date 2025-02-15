# BlackHoleRender

CompSci Masters project of an existing project to render a black hole.

## Compile

`g++ -O3 Functions.cpp main.cpp -o Executable -lpng -fopenmp`

## Run

On a Linux terminal, use input parameters to specify custom configurations. Otherwise, the system will show the inputs as a note, and run with default values

> note: keep the image dimension and ray dimension to a 1:4 ratio, since the anti-alias does area-sampling and squeezes the output image to 25%


`./Executable 1024 1024 256 256`

## Tasks

Scope No1: 
- Report and Code: 
  - optimised for multiple cores of CPUs AND for GPUs.
  - distribute the program.

Scope No2: 
- Contribute to the project:
  - enhance the program

Refactor Eigen library use, instead create own functions and methods.
Use parameters through the code.

Additional task: profile the code, to see how much of: RAM, Cache, and Registers are used.

## Contributors

J. Gergely *Gary* Hornyak
