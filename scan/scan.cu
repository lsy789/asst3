#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <driver_functions.h>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

#include "CycleTimer.h"

#define THREADS_PER_BLOCK 256


// helper function to round an integer up to the next power of 2
static inline int nextPow2(int n) {
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

__global__ void
up_sweep(int N, int* device_result, int twod)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int two_dplus1 = 2*twod;

    device_result[(index+1) * two_dplus1 -1] += device_result[index * two_dplus1 + twod -1];
}

__global__ void
down_sweep(int N, int* device_result, int twod)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int two_dplus1 = 2*twod;

    if (((index+1) * two_dplus1 -1) < N) {
        int tmp = device_result[index*two_dplus1+twod-1];
        int tmp1 = device_result[(index+1)*two_dplus1-1];
        device_result[index*two_dplus1+twod-1] = tmp1;
        device_result[(index+1)*two_dplus1-1] = tmp1 + tmp;
    }
    
}

__global__ void
zero(int N, int* device_result)
{
    device_result[N-1] = 0;
}

// exclusive_scan --
//
// Implementation of an exclusive scan on global memory array `input`,
// with results placed in global memory `result`.
//
// N is the logical size of the input and output arrays, however
// students can assume that both the start and result arrays we
// allocated with next power-of-two sizes as described by the comments
// in cudaScan().  This is helpful, since your parallel scan
// will likely write to memory locations beyond N, but of course not
// greater than N rounded up to the next power of 2.
//
// Also, as per the comments in cudaScan(), you can implement an
// "in-place" scan, since the timing harness makes a copy of input and
// places it in result
void exclusive_scan(int* input, int N, int* result)
{

    // CS149 TODO:
    //
    // Implement your exclusive scan implementation here.  Keep in
    // mind that although the arguments to this function are device
    // allocated arrays, this is a function that is running in a thread
    // on the CPU.  Your implementation will need to make multiple calls
    // to CUDA kernel functions (that you must write) to implement the
    // scan.
    int *tmp = new int[N];
    int d = N/2;
    int twod = 1;
    while (d > THREADS_PER_BLOCK) {
        int block_size = d / THREADS_PER_BLOCK;
        up_sweep<<<block_size, THREADS_PER_BLOCK>>>(N, result, twod);
        d = d / 2;
        twod = twod * 2;
    }

    while (d > 0) {
        up_sweep<<<1, d>>>(N, result, twod);
        d = d / 2;
        twod = twod * 2;
    }

    zero<<<1, 1>>>(N, result);

    // downsweep phase
    d = 1;
    twod = N / 2;
    while (d <= THREADS_PER_BLOCK) {
         down_sweep<<<1, d>>>(N, result, twod);
         twod = twod / 2;
         d = d * 2;
    }

    while (d < N) {
        int block_size = d / THREADS_PER_BLOCK;
        down_sweep<<<block_size, THREADS_PER_BLOCK>>>(N, result, twod);
        twod = twod / 2;
        d = d * 2;
    }

}


//
// cudaScan --
//
// This function is a timing wrapper around the student's
// implementation of scan - it copies the input to the GPU
// and times the invocation of the exclusive_scan() function
// above. Students should not modify it.
double cudaScan(int* inarray, int* end, int* resultarray)
{
    int* device_result;
    int* device_input;
    // int N = end - inarray;  

    // This code rounds the arrays provided to exclusive_scan up
    // to a power of 2, but elements after the end of the original
    // input are left uninitialized and not checked for correctness.
    //
    // Student implementations of exclusive_scan may assume an array's
    // allocated length is a power of 2 for simplicity. This will
    // result in extra work on non-power-of-2 inputs, but it's worth
    // the simplicity of a power of two only solution.

    int rounded_length = nextPow2(end - inarray);
    
    cudaMalloc((void **)&device_result, sizeof(int) * rounded_length);
    cudaMalloc((void **)&device_input, sizeof(int) * rounded_length);

    // For convenience, both the input and output vectors on the
    // device are initialized to the input values. This means that
    // students are free to implement an in-place scan on the result
    // vector if desired.  If you do this, you will need to keep this
    // in mind when calling exclusive_scan from find_repeats.
    cudaMemcpy(device_input, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_result, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    exclusive_scan(device_input, rounded_length, device_result);

    // Wait for completion
    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();
       
    cudaMemcpy(resultarray, device_result, (end - inarray) * sizeof(int), cudaMemcpyDeviceToHost);

    double overallDuration = endTime - startTime;
    cudaFree(device_result);
    cudaFree(device_input);
    return overallDuration; 
}


// cudaScanThrust --
//
// Wrapper around the Thrust library's exclusive scan function
// As above in cudaScan(), this function copies the input to the GPU
// and times only the execution of the scan itself.
//
// Students are not expected to produce implementations that achieve
// performance that is competition to the Thrust version, but it is fun to try.
double cudaScanThrust(int* inarray, int* end, int* resultarray) {

    int length = end - inarray;
    thrust::device_ptr<int> d_input = thrust::device_malloc<int>(length);
    thrust::device_ptr<int> d_output = thrust::device_malloc<int>(length);
    
    cudaMemcpy(d_input.get(), inarray, length * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    thrust::exclusive_scan(d_input, d_input + length, d_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();
   
    cudaMemcpy(resultarray, d_output.get(), length * sizeof(int), cudaMemcpyDeviceToHost);

    thrust::device_free(d_input);
    thrust::device_free(d_output);

    double overallDuration = endTime - startTime;
    return overallDuration; 
}
__global__ void
write_flag(int* device_input, int* device_flag, int* device_index, int N) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < N-1 && device_input[index] == device_input[index+1]) {
        device_flag[index] = 1;
        device_index[index] = 1;
    }

    if (index < N-1 && device_input[index] != device_input[index+1]) {
        device_flag[index] = 0;
        device_index[index] = 0;
    }

}

__global__ void
print_repeat(int* device_index, int* device_flag, int* device_output, int N) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < N && device_flag[index] == 1) {
        device_output[device_index[index]] = index;
    }
}


// find_repeats --
//
// Given an array of integers `device_input`, returns an array of all
// indices `i` for which `device_input[i] == device_input[i+1]`.
//
// Returns the total number of pairs found
int find_repeats(int* device_input, int length, int* device_output) {

    // CS149 TODO:
    //
    // Implement this function. You will probably want to
    // make use of one or more calls to exclusive_scan(), as well as
    // additional CUDA kernel launches.
    //    
    // Note: As in the scan code, the calling code ensures that
    // allocated arrays are a power of 2 in size, so you can use your
    // exclusive_scan function with them. However, your implementation
    // must ensure that the results of find_repeats are correct given
    // the actual array length.
    int *device_flag;
    int *device_index;
    int *device_count;
    int count;
    int rounded_length = nextPow2(length);

    cudaMalloc((void **)&device_flag, rounded_length * sizeof(int));
    cudaMalloc((void **)&device_index, rounded_length * sizeof(int));
    cudaMalloc((void **)&device_count, 1 * sizeof(int));

    int block_size = length % THREADS_PER_BLOCK !=0 ? length / THREADS_PER_BLOCK+1 : length / THREADS_PER_BLOCK;
    write_flag<<<block_size, THREADS_PER_BLOCK>>>(device_input, device_flag, device_index, length);

    exclusive_scan(device_index, rounded_length, device_index);

    print_repeat<<<block_size, THREADS_PER_BLOCK>>>(device_index, device_flag, device_output, length);

    cudaMemcpy(&count, device_index + length -1, 1* sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(device_flag);
    cudaFree(device_index);
    cudaFree(device_count);

    return count; 
}


//
// cudaFindRepeats --
//
// Timing wrapper around find_repeats. You should not modify this function.
double cudaFindRepeats(int *input, int length, int *output, int *output_length) {

    int *device_input;
    int *device_output;
    int rounded_length = nextPow2(length);
    
    cudaMalloc((void **)&device_input, rounded_length * sizeof(int));
    cudaMalloc((void **)&device_output, rounded_length * sizeof(int));
    cudaMemcpy(device_input, input, length * sizeof(int), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    double startTime = CycleTimer::currentSeconds();
    
    int result = find_repeats(device_input, length, device_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();

    // set output count and results array
    *output_length = result;
    cudaMemcpy(output, device_output, length * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(device_input);
    cudaFree(device_output);

    float duration = endTime - startTime; 
    return duration;
}



void printCudaInfo()
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++)
    {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n"); 
}
