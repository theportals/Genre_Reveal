# Methodology Descriptions
## Serial
For the serial k-means clustering, we adapted the code from [this tutorial](http://reasonabledeviations.com/2019/10/02/k-means-in-cpp/) to work with 3 dimensions instead of 2. In essence, the algorithm:
1. Randomly selects `k` points to act as "centroids"
2. Loops through each point, assigning it to the nearest centroid
3. Finds the average x, y, and z coordinate of each cluster
4. Moves each centroid to be in the center of its cluster
5. Repeats steps 2-4 until it converges (i.e, the centroids move an insignificant amount)

## Parallel Shared Memory CPU
To parallelize with shared memory, we used OpenMP to spawn multiple threads to handle steps 2, 3, and 4 of the k-means algorithm. Essentially, each thread works on one point at a time, updating data in shared memory, until all points have been accounted for.

## Parallel Shared Memory GPU
The shared memory GPU implementation works similarly to the shared memory CPU implementation, using a kernel function to parallelize steps 2, 3, and 4, and updates the values in shared memory.

## Distributed Memory CPU
The distributed memory CPU implementation uses MPI (Message Passing Interface). In this implementation, each thread is given a roughly equal number of points. Each thread then performs step 2 on its own set of data, and adds the x, y, and z coordinate to a global array. Thread 0 then divides the global sums to finish step 3, performs step 4, and if the centroids have not converged, sends the updated centroids out to each thread.  

## Distributed Memory GPU
The distributed memory GPU implementation works very similarly to the distributed memory CPU implementation, only with each thread accessing its available GPU to assign each point to a centroid.