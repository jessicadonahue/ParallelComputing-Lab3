#include <stdio.h>
#include <locale.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>


long getmax(long *, long);



__global__ void getMaxNum( long *in, long size, long blocks_d, long *out) {

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x *blockDim.x + threadIdx.x;

    // each thread loads one element from global to shared mem
    long x = LONG_MIN; 
    if(i < size) 
        x = in[i]; 
    out[blockIdx.x*blocks_d+tid] = x;

    __syncthreads();


    for (unsigned int s=blockDim.x/2; s>0; s = s/2){
        if (tid < s){

            __syncthreads();

            if (out[blockIdx.x * blocks_d+tid] < out[blockIdx.x * blocks_d+tid+s]){
                out[blockIdx.x * blocks_d+tid] = out[blockIdx.x * blocks_d+tid+s];
            }
           
        }
        __syncthreads();
    }

    __syncthreads();



}


int main(int argc, char *argv[])
{
    long size = 0;  // The size of the array
    long i;  // loop index
    long max;
    long * numbers; //pointer to the array

    if(argc !=2)
    {
        printf("usage: maxseq num\n");
        printf("num = size of the array\n");
        exit(1);
    }

    size = atol(argv[1]);

    numbers = (long *)malloc(size * sizeof(long));
    if( !numbers )
    {
        printf("Unable to allocate mem for an array of size %ld\n", size);
        exit(1);
    }    

    srand(time(NULL)); // setting a seed for the random number generator
    // Fill-up the array with random numbers from 0 to size-1 
    for( i = 0; i < size; i++){
        numbers[i] = rand() % size; 
    }

   
    int numsSize = size * sizeof(long);


    // Get the number of threads per block but getting the device's
    // maximum threads per block
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, 0);
    long THREADS_PER_BLOCK = devProp.maxThreadsPerBlock;

    //Create nums array that we will be sending to the device
    long * nums;

    // Get number of blocks by rounding up the size of the array / threads per block
    // so, the amount of blocks needed for the max threads per block for this device
    long blocks = ((size + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK);
    long numOfThreads;
    if (size > THREADS_PER_BLOCK){
        numOfThreads = THREADS_PER_BLOCK;
    }
    else{
        numOfThreads = size;
    }

    
    //Transfer and copy numbers array from the host to the device
    cudaMalloc((void **) &nums, numsSize);

    cudaError_t e = cudaMemcpy(nums, numbers, numsSize, cudaMemcpyHostToDevice);

    
    // Create array that will store the result - sending this from device to host
    long * maxResult;
    long resultSize = blocks * sizeof(long);

    // Transfer maxResult array to device
    cudaMalloc((void **) &maxResult, resultSize);
    cudaError_t v  = cudaGetLastError();

    //launch kernel function
    getMaxNum<<<blocks, numOfThreads>>>(nums, size, blocks, maxResult);

    // Copy the array from the device to the host so we can get result
    cudaError_t s = cudaMemcpy(numbers, maxResult, resultSize, cudaMemcpyDeviceToHost);

    long l;
    max = numbers[0];

    for(l = 1; l < blocks; l++){
        if(numbers[l] > max){
            max = numbers[l];
        }
    }

    printf("The maximum number in the array is %'ld \n", max);


    cudaFree(nums);
    cudaFree(maxResult);
    free(numbers);
    cudaDeviceReset();
    exit(0);
}
