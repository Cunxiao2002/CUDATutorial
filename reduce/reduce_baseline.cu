#include<stdio.h>
#include<cuda.h>
#include "cuda_runtime.h"

//reduce_baseline:1733.15ms

__global__ void reduce_baseline(const int* input, int* output, size_t n){

  int sum = 0;
  for(size_t i = 0; i < n; i ++ ) {
    sum += input[i];
  }

  *output = sum;

}

bool CheckResult(int *out, int groudtruth, int n){
  if(*out != groudtruth){
    return false;
  } else {
      return true;
  }
}

int main() {
    float millseconds = 0;
    const int N = 25600000;
    
    cudaSetDevice(7);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    const int blocksize = 1;
    int gridsize = 1;
    
    int *a = (int *)malloc(N * sizeof(int));
    int *d_a;
    cudaMalloc((void**)&d_a, N * sizeof(int));

    int *out = (int*)malloc(gridsize * sizeof(int));
    int *d_out;
    cudaMalloc((void**)&d_out, gridsize * sizeof(int));

    for(int i = 0; i < N; i ++ ){
        a[i] = 1;
    }

    int groudtruth = N * 1;
    cudaMemcpy(d_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
    dim3 Grid(gridsize);
    dim3 Block(blocksize);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    reduce_baseline<<<1, 1>>> (d_a, d_out, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&millseconds, start, stop);

    cudaMemcpy(out, d_out, gridsize * sizeof(int), cudaMemcpyDeviceToHost);
    
    bool is_right = CheckResult(out, groudtruth, gridsize);
    
    if(is_right){
        printf("the ans is right\n");
    }else{
        printf("the ans is wrong\n");
        for(int i = 0; i < gridsize; i ++ ){
            printf("res per block : %f", out[i]);
        }
        printf("\n");
        printf("groudtruth is %f \n", groudtruth);
    }
    
    printf("reduce_base latency = %f ms\n", millseconds);

    cudaFree(d_a);
    cudaFree(d_out);
    free(a);
    free(out);
}
