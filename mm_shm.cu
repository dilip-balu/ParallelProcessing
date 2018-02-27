#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include <sys/time.h>
#include <time.h>

#define A_WIDTH 1024
#define A_HEIGHT 1024
#define B_WIDTH 1024
#define B_HEIGHT 1024
#define C_WIDTH B_WIDTH
#define C_HEIGHT A_HEIGHT

#define BLOCK_SIZE 16
#define NUM_SUBS (A_WIDTH / BLOCK_SIZE)

__device__ float d_A[A_HEIGHT][A_WIDTH];
__device__ float d_B[B_HEIGHT][B_WIDTH];
__device__ float d_C[C_HEIGHT][C_WIDTH];

float h_A[A_HEIGHT][A_WIDTH];
float h_B[B_HEIGHT][B_WIDTH];
float h_C[C_HEIGHT][C_WIDTH];
float h_C_ref[C_HEIGHT][C_WIDTH];

void checkCUDAError(const char *msg);
void matrixMulCPU(float A[A_HEIGHT][A_WIDTH], float B[B_HEIGHT][B_WIDTH], float C[C_HEIGHT][C_WIDTH]);
int matrixMulTest(float C[C_HEIGHT][C_WIDTH], float Cref[C_HEIGHT][C_WIDTH]);

__global__ void matrixMulCUDA()
{
	// Block index
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int x = bx*BLOCK_SIZE + tx;
	int y = by*BLOCK_SIZE + ty;

	float Csub = 0;
        __shared__ float AS[BLOCK_SIZE][BLOCK_SIZE+1];
        __shared__ float BS[BLOCK_SIZE][BLOCK_SIZE+1];
        int NITER = A_WIDTH/BLOCK_SIZE;
         
        for(int i=0;i<NITER;i++){
     
            int a_y = (by*BLOCK_SIZE)+ty;
            int a_x = (i*BLOCK_SIZE)+tx;
            int b_y = (i*BLOCK_SIZE)+ty;
            int b_x = (bx*BLOCK_SIZE)+tx;   

            AS[ty][tx] = d_A[a_y][a_x];
            BS[ty][tx] = d_B[b_y][b_x];

             
           __syncthreads();

        
	//iterate A_WIDTH (same as B_HEIGHT) to calculate the product
	for (int k = 0; k < BLOCK_SIZE; k++){
		Csub += AS[ty][k] * BS[k][tx]; 
	}
          
        __syncthreads();
	// Store the product value of C matrix
	d_C[y][x] = Csub;

        } 

           
}


int main(int argc, char **argv)

{
	unsigned int mem_size_A, mem_size_B, mem_size_C;
	unsigned int x, y, errors;
	float msec;
	cudaEvent_t start, stop;

	if (A_WIDTH != B_HEIGHT){
		printf("Error: A_HEIGHT and B_WIDTH do not match\n");
	}

	mem_size_A = sizeof(float)* A_WIDTH* A_HEIGHT;
	mem_size_B = sizeof(float)* B_WIDTH* B_HEIGHT;
	mem_size_C = sizeof(float)* C_WIDTH* C_HEIGHT;

	// Initialise A
	for (y = 0; y < A_HEIGHT; y++)
	    for (x = 0; x <A_WIDTH; x++)
		h_A[y][x] = (float)rand() / RAND_MAX;
	// Initialise B
	for (y = 0; y < B_HEIGHT; y++)
	    for (x = 0; x <B_WIDTH; x++)
		h_B[y][x] = (float)rand() / RAND_MAX;

	// copy host memory to device
	cudaMemcpyToSymbol(d_A, h_A, mem_size_A);
	cudaMemcpyToSymbol(d_B, h_B, mem_size_B);
	checkCUDAError("CUDA memcpy");

	// Allocate CUDA events that we'll use for timing
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	checkCUDAError("CUDA event creation");

	// Setup execution parameters
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid(C_WIDTH / BLOCK_SIZE, C_HEIGHT / BLOCK_SIZE);
	cudaEventRecord(start);
	matrixMulCUDA << < grid, threads >> >();
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	checkCUDAError("CUDA kernel execution and timing");

	cudaEventElapsedTime(&msec, start, stop);
	cudaThreadSynchronize();
	checkCUDAError("CUDA timing");

	// Copy result from device to host
	cudaMemcpyFromSymbol(h_C, d_C, mem_size_C);
	checkCUDAError("CUDA memcpy results");

	// Compute reference CPU version
	float elapsed_ms ;
	struct timeval  start_time , stop_time;
	gettimeofday (&start_time , (struct timezone *)NULL);
	matrixMulCPU(h_A, h_B, h_C_ref);
	gettimeofday (&stop_time , (struct timezone *)NULL);
	elapsed_ms = ((stop_time.tv_sec - start_time.tv_sec) +
	     (stop_time.tv_usec - start_time.tv_usec)/1000000.0) * 1000.0 ;


	// Check for errors
	errors = matrixMulTest(h_C, h_C_ref);
	if (errors)
		printf("%d total errors\n", errors);
	else
		printf("Test passed successfully\n");

	printf("Kernel time: %f ms, CPU time: %f ms\n", msec, elapsed_ms );

}


void matrixMulCPU(float A[A_HEIGHT][A_WIDTH], float B[C_HEIGHT][C_WIDTH], float C[C_HEIGHT][C_WIDTH])
{
	int x, y, k;
	for (y = 0; y < C_HEIGHT; y++){
		for (x = 0; x < C_WIDTH; x++){
			C[y][x] = 0;
			for (k = 0; k < A_WIDTH; k++){
				C[y][x] += A[y][k] * B[k][x];
			}
		}
	}

}

int matrixMulTest(float C[C_HEIGHT][C_WIDTH], float Cref[C_HEIGHT][C_WIDTH])
{
	int errors = 0;
	int y, x;

        double eps = 1.e-6 ; 
	for (y = 0; y < C_HEIGHT; y++){
		for (x = 0; x < C_WIDTH; x++){
			if (fabs(C[y][x] - Cref[y][x])/fabs(C[y][x])/A_WIDTH > eps ){
				errors++;
				printf("Device item c[%d][%d] = %f does not mach host result %f\n", y, x, C[y][x], Cref[y][x]);
			}
		}
	}

	return errors;
}

void checkCUDAError(const char *msg)
{
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err)
	{
		fprintf(stderr, "CUDA ERROR: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}
