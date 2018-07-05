
#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <ctime>
#include <random> // for c++11 random number generation on host
#include <cstdio>
#include <cstdlib> // for min/max
#include <curand_kernel.h> // random number
#include <device_functions.h>
#include <time.h>
#include <fstream>
#include <string>
#include <numeric>
#include <curand.h>


#define N 500 // curand_state objects or grid size

using namespace std;

//debug outputs
#define CUDA_KERNEL_DEBUG 0 //test for illegal memory access
#define OUTPUT_PRE 1 // preprocess debug output
#define OUTPUT_POST 1 //postprocess debug output

// Error wrapper
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort = true)
{
	if (code != cudaSuccess) {
		std::cout << "GPUassert: " << cudaGetErrorString(code) << " / " << file << " " << line << std::endl;
		//fcout << stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
	else {
		if (CUDA_KERNEL_DEBUG == 1) {
			std::cout << "success GPUassert: " << cudaGetErrorString(code) << " / " << file << " " << line << std::endl;
		}
	}
}

//initialize random numbers
__device__ void init( curandState_t* states) {

	int Idx = blockIdx.x *blockDim.x + threadIdx.x; // each core has threads of random numbers
	curand_init(clock(), Idx, 0, &states[Idx]);

}

//__device__ float curand_uniform(curandState_t *states) {
//
//	return curand_uniform(&states);
//
//};

__global__ void randoms( curandState_t* states, float* num)
{
	int Idx = blockIdx.x *blockDim.x + threadIdx.x;
	init(states);
	num[Idx] = curand_uniform(&states[Idx]);
}

int main()
{
	dim3 dimblock; //threads
	dim3 dimgrid; // blocks
	curandState_t* states;
	cudaMalloc((void**)&states, N * sizeof(curandState_t));
		float cpu_nums[N];
		float* gpu_nums;
	cudaMalloc((void**) &gpu_nums, N * sizeof(float));

	//invoke random number kernel
	dimgrid.x = N / 2; dimgrid.y = N / 2; dimgrid.z = 1; // grid of blocks
	dimblock.x = (N / 2 + (N / 2 - 1)) / (N / 2); dimblock.y = (N / 2 + (N / 2 - 1)) / (N / 2); dimblock.z = 1; // block of threads
																												// allocate array of ints on CPU and GPUs
	randoms << < dimgrid, dimblock >> > (states, gpu_nums);
	cudaMemcpy(cpu_nums, gpu_nums, N * sizeof(float), cudaMemcpyDeviceToHost);

	for (int i = 0; i < N; i++) {

		cout << "the " << i <<  " numbers is: " << cpu_nums[i] << endl;
	}

	cudaFree(states);
	cudaFree(gpu_nums);

	cin.get();
	return 0;

}