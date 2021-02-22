#include <stdio.h>
#include <stdlib.h>
#include <math.h>


// CUDA kernel. Each thread takes care of one element of c
// threadIdx.x gives thread id
__global__ void vecAdd(double *a, double *b, double *c, int n)
{
// Get our global thread ID
int id = threadIdx.x;
// Make sure we do not go out of bounds
if (id < n)
c[id] = a[id] + b[id];
}


int main( int argc, char* argv[] ) {
int n = 1000; // Size of vectors
int i;
double *h_a, *h_b; // input vectors
double *h_c; // output vector
size_t bytes = n*sizeof(double); // Size, in bytes, of each vector
// Allocate memory for each vector on host
h_a = (double*)malloc(bytes); h_b = (double*)malloc(bytes); h_c = (double*)malloc(bytes);
// Initialize vectors on host
for( i = 0; i < n; i++ ) { h_a[i] = rand(); h_b[i] = rand(); }
double *d_a, *d_b; // Device input vectors
double *d_c; //Device output vector
// Allocate memory for each vector on GPU
cudaMalloc(&d_a, bytes); cudaMalloc(&d_b, bytes); cudaMalloc(&d_c, bytes);
// Copy data into device (GPU) memory
cudaMemcpy( d_a, h_a, bytes, cudaMemcpyHostToDevice);
cudaMemcpy( d_b, h_b, bytes, cudaMemcpyHostToDevice);
// Launch kernels
vecAdd<<<1, n>>>(d_a, d_b, d_c, n);
// Copy output data into Host memory
cudaMemcpy( d_a, h_a, bytes, cudaMemcpyDeviceToHost);
cudaMemcpy( d_b, h_b, bytes, cudaMemcpyDeviceToHost);
// Free device memory
cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
// WE ARE DONE – back in Host (CPU) processing
// Free memory
free(h_a); free(h_b); free(h_c);
free(d_a); free(d_b); free(d_c);
return 0;
}
