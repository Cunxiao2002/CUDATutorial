#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include <cuda.h>

//v5:展开for循环
//latency:0.3939ms

template <int blocksize>
__device__ void BlockReduce(float* sdata) {
  if(blocksize >= 1024) {
    if(threadIdx.x < 512) {
      sdata[threadIdx.x] += sdata[threadIdx.x + 512];
    }
    __syncthreads();
  }

  if(blocksize >= 512) {
    if(threadIdx.x < 256) {
      sdata[threadIdx.x] += sdata[threadIdx.x + 256];
    }
    __syncthreads();
  }

  if(blocksize >= 256) {
    if(threadIdx.x < 128) {
      sdata[threadIdx.x] += sdata[threadIdx.x + 128];
    }
    __syncthreads();
  }

  if(blocksize >= 128) {
    if(threadIdx.x < 64) {
      sdata[threadIdx.x] += sdata[threadIdx.x + 64];
    }
    __syncthreads();
  }

  /*  
  if(blocksize >= 64) {
    if(threadIdx.x < 32) {
      sdata[threadIdx.x] = sdata[threadIdx.x + 32];
    }
    __syncthreads();
  }
  */

  if(threadIdx.x < 32) {
    volatile float* vshm = sdata;
    if(blockDim.x >= 64) {
      vshm[threadIdx.x] += vshm[threadIdx.x + 32];
    }
    vshm[threadIdx.x] += vshm[threadIdx.x + 16];
    vshm[threadIdx.x] += vshm[threadIdx.x + 8];
    vshm[threadIdx.x] += vshm[threadIdx.x + 4];
    vshm[threadIdx.x] += vshm[threadIdx.x + 2];
    vshm[threadIdx.x] += vshm[threadIdx.x + 1];
  }
}

template<int blocksize>
__global__ void reduce_v5(float *d_in, float *d_out) {
  int tid = threadIdx.x;
  //int gtid = blockDim.x * blockIdx.x + threadIdx.x;
  int gtid = blockDim.x * blockIdx.x * 2 + threadIdx.x;

  __shared__ float smem[blocksize];

  //load data to smem
  //smem[tid] = d_in[gtid];
  smem[tid] = d_in[gtid] + d_in[gtid + blocksize];
  __syncthreads();

  //compute
  BlockReduce<blocksize> (smem);
  
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
  }
  return true;
}

int main() {
  //定义设备信息
  cudaSetDevice(0);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);

  //定义参数：gridsize、blocksize
  const int N = 25600000;
  const int blocksize = 256;
  int gridsize = std::min((N + blocksize - 1) / blocksize, deviceProp.maxGridSize[0]); 
  dim3 Grid(gridsize);
  dim3 Block(blocksize / 2);

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
  reduce_v5<blocksize / 2> <<<Grid, Block>>> (d_a, d_out);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&millisecond, start, stop);

  cudaMemcpy(out, d_out, gridsize * sizeof(float), cudaMemcpyDeviceToHost);

  //验证正确性
  bool is_right = CheckResult(out, groudtruth, gridsize);
  if(is_right) {
    printf("the anwser is right\n");
  } else {
    printf("the anwser is wrong\n");
    //printf("the wrong anwser is %.2f\n", out[0]);
    printf("the groudtruth is %.2f\n", groudtruth);
  }
  printf("reduce_v5 latency = %.6f ms \n", millisecond);

  //释放内存
  cudaFree(d_a);
  cudaFree(d_out);
  free(a);
  free(out);
}