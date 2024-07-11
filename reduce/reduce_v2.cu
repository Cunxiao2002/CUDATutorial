#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include <cuda.h>

//v2:改变reduce的方式，原本是紧邻的两个thread进行reduce，改成前一半和后一半进行reduce
//latency:0.502ms
template<int blocksize>
__global__ void reduce_v1(float *d_in, float *d_out) {
  int tid = threadIdx.x;
  int gtid = blockDim.x * blockIdx.x + threadIdx.x;

  __shared__ float smem[blocksize];

  //load data to smem
  smem[tid] = d_in[gtid];
  __syncthreads();

  //compute
  for(int i = blockDim.x / 2; i > 0; i /= 2) {
    if(tid < i) { 
      smem[tid] += smem[tid + i];
    }
    __syncthreads();
  }

  //store
  if(tid == 0){
    d_out[blockIdx.x] = smem[0];
  }
}

bool CheckResult(float* out, float groudtruth, int n) {
  float res = 0.0;
  for(int i = 0; i < n; i ++ ) {
    res += out[i];
  }
  if(res != groudtruth) {
    return false;
  } else {
    return true;
  }
}

int main() {
  //定义设备信息
  cudaSetDevice(0);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);

  //定义参数：gridsize、blocksize
  const int N = 25600000;
  const int blocksize = 256;
  int gridsize = std::min((N + blocksize - 1) / blocksize,deviceProp.maxGridSize[0]); 
  dim3 Grid(gridsize);
  dim3 Block(blocksize);

  //定义变量并分配内存
  float *a = (float *)malloc(N * sizeof(float));
  float *d_a;
  cudaMalloc((void **)&d_a, N * sizeof(float));
  

  float *out = (float *)malloc(gridsize * sizeof(float));
  float *d_out;
  cudaMalloc((void **)&d_out, gridsize * sizeof(float));

  //初始化输入数据
  for(int i = 0; i < N; i ++ ){
    a[i] = 1.0f;
  }

  float groudtruth = N * 1.0f;
  //创建事件，对kernel计时
  cudaMemcpy(d_a, a, N * sizeof(float), cudaMemcpyHostToDevice);

  float millisecond = 0;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  reduce_v1<blocksize> <<<Grid, Block>>> (d_a, d_out);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&millisecond, start, stop);

  cudaMemcpy(out, d_out, gridsize * sizeof(float), cudaMemcpyDeviceToHost);

  //验证正确性
  bool is_right = CheckResult(out, groudtruth, gridsize);
  if(is_right) {
    printf("the anwser is right\n");
  } else {
    //printf("the anwser is wrong\n");
    printf("the wrong anwser is %.2f\n", out[0]);
    printf("the groudtruth is %.2f\n", groudtruth);
  }
  printf("reduce_v2 latency = %.6f ms \n", millisecond);

  //释放内存
  cudaFree(d_a);
  cudaFree(d_out);
  free(a);
  free(out);
}