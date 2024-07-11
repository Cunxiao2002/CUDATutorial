#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include <cuda.h>

//v4:展开for循环最后一个warp
//latency:0.4097ms

//展开for循环最后一个warp
__device__ void WarpSharedMemReduce(volatile float* smem, int tid) {
  float x = smem[tid];
  if(blockDim.x >= 64){
    x += smem[tid + 32];
    __syncwarp();
    smem[tid] = x; 
    __syncwarp();
  }

  x += smem[tid + 16];
  __syncwarp();
  smem[tid] = x;
  __syncwarp();

  x += smem[tid + 8];
  __syncwarp();
  smem[tid] = x;
  __syncwarp();

  x += smem[tid + 4];
  __syncwarp();
  smem[tid] = x;
  __syncwarp();

  x += smem[tid + 2];
  __syncwarp();
  smem[tid] = x;
  __syncwarp();

  x += smem[tid + 1];
  __syncwarp();
  smem[tid] = x;
  __syncwarp();

}

template<int blocksize>
__global__ void reduce_v4(float *d_in, float *d_out) {
  int tid = threadIdx.x;
  //int gtid = blockDim.x * blockIdx.x + threadIdx.x;
  int gtid = blockDim.x * blockIdx.x * 2 + threadIdx.x;

  __shared__ float smem[blocksize];

  //load data to smem
  //smem[tid] = d_in[gtid];
  smem[tid] = d_in[gtid] + d_in[gtid + blocksize];
  __syncthreads();

  //compute
  for(int i = blockDim.x / 2; i > 32; i >>= 1) {
    if(tid < i) { 
      smem[tid] += smem[tid + i];
    }
    __syncthreads();
  }

  if(tid < 32){
    WarpSharedMemReduce(smem, tid);
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
  reduce_v4<blocksize / 2> <<<Grid, Block>>> (d_a, d_out);
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
  printf("reduce_v4 latency = %.6f ms \n", millisecond);

  //释放内存
  cudaFree(d_a);
  cudaFree(d_out);
  free(a);
  free(out);
}