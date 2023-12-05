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
The files `dist_gpu.cpp` and `updateCentroids` must be built using the `-c` flag, and then linked. Some systems may require modules to be loaded, as well. Example:
```bash
module load gcc[/version] mpi cuda
mpicxx -c dist_gpu.cpp -o dist_gpu.o
nvcc -c updateCentroids.cu -o updateCentroids.o
mpicxx dist_gpu.o updateCentroids.o -lcudart -o <output file>
```