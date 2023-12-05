# Building separately
## Serial
Our CSV parser requires pthreads to be linked to the compiler. g++ example:
```bash
g++ -pthread -o <output file> serial.cpp
```

## Shared CPU
pthreads and openmp must be linked to the compiler. g++ example:
```bash
g++ -pthread -fopenmp -o <output file> shared_cpu.cpp
```

## Shared GPU
This must be compiled with `nvcc`, and you may have to specify the Cuda architecture. Example:
```bash
nvcc -arch=sm_70 -o <output file> shared_gpu.cu
```

## Distributed CPU
To use MPI, you must use an MPI C++ compiler. Example:
```bash
mpicxx -o <output file> dist_cpu.cpp
```

## Distributed GPU
WIP