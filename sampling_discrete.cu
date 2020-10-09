// experiments with different versions of sampling from a discrete distribution

// compilation and execution commands:
// nvcc -arch=sm_35 -rdc=true sampling_discrete.cu
// ./a.out input.txt > sampling_discrete_outputs.txt

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <cuda.h>

#define TRIALS 100
#define K_MIN 8
#define K_MAX 1024

__global__ void prefix_linear_kernel(double* dp, double u, int* dj, int K) {
  int i;

  // computing prefix sum
  double sum = 0.0;
  for (i = 0; i < K; i++) {
    sum += dp[i];
    dp[i] = sum;
  }

  // cumulative probability to be searched for
  double u1 = u * sum;

  // linear search on prefix sum array
  i = 0;
  while (i < K-1 && u1 >= dp[i]) {
    i++;
  }

  // storing the sampled index
  *dj = i;
}

__global__ void prefix_binary_kernel(double* dp, double u, int* dj, int K) {
  int i, k, mid;

  // computing prefix sum
  double sum = 0.0;
  for (i = 0; i < K; i++) {
    sum += dp[i];
    dp[i] = sum;
  }

  // cumulative probability to be searched for
  double u1 = u * sum;

  // binary search on prefix sum array
  i = 0;
  k = K-1;
  while (i < k) {
    mid = (i + k)/2;
    if (u1 < dp[mid]) {
      k = mid;
    }
    else {
      i = mid + 1;
    }
  }

  // storing the sampled index
  *dj = i;
}

__global__ void construct_partial_sum_array_kernel(double* dp, int b) {
  int i = threadIdx.x + 1;

  // a step in the parallel partial sum calculation
  dp[(2*i * ((int) pow(2, b-1))) - 1] += dp[((2*i - 1) * ((int) pow(2, b-1))) - 1];
}

__global__ void partial_binary_kernel(double* dp, double u, int* dj, int K, int log_K) {
  int i, b, k, mid;
  // computing partial sums
  for (b = 1; b <= log_K; b++) {
    int ub = (int) pow(2, (log_K - b));

    // dynamic parallelism in action
    construct_partial_sum_array_kernel<<<1, ub>>>(dp, b);

    // to synchronize between the child threads and the parent thread
    cudaDeviceSynchronize();
  }

  // cumulative probability to be searched for
  double u1 = u * dp[K - 1];

  // modified binary search on partial sum array
  i = 0;
  k = K - 1;
  double lowValue = 0.0;
  double compareValue;
  while (i < k) {
    mid = (i + k)/2;
    compareValue = lowValue + dp[mid];
    if (u1 < compareValue) {
      k = mid;
    }
    else {
      i = mid + 1;
      lowValue = compareValue;
    }
  }

  // storing the sampled index
  *dj = i;
}

__global__ void prefix_sum_kernel(double* dp, int K) {
  int uid = threadIdx.x;

  // one step in the parallel computation of prefix sums
  int off;
  double temp;

  for (off = 1; off < K; off *= 2) {
    // first we perform the reads
    if (threadIdx.x >= off) {
        temp = dp[uid - off];
    }
    __syncthreads();

    // then we perform the writes
    if (threadIdx.x >= off) {
        dp[uid] += temp;
    }
    __syncthreads();
  }
}

__global__ void sample_element_kernel(double* dp, int* dj, int K, double u) {
  int uid = threadIdx.x;

  // parallel search of the prefix sum array for sampling one of the elements
  double u1 = u * dp[K-1];
  if (u1 >= dp[uid]) {
    atomicMax(dj, uid);
  }
}

// function to read the relative probabilities array from the input file
void initArray(double* a, int K, char* filename) {
  int i;
  FILE* f = fopen(filename, "r");
  for (i = 0; i < K; i++) {
    fscanf(f, "%lf", &a[i]);
  }
  fclose(f);
}

// takes as command-line argument the path to the input file
int main(int argc, char* argv[]) {
  srand(time(NULL));

  // number of elements to be sampled from
  int K;

  // to be used in partial sum calculation
  int log_K;

  // creating GPU memory for the index sampled
  int* dj;
  cudaMalloc(&dj, sizeof(int));

  // to be used in random sampling
  double u;

  // to time the execution of kernels
  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  float elapsed, elapsed1, elapsed2, elapsed3, elapsed4;

  int i, iter;
  printf("K,t1,t2,t3,t4\n");
  for (i = K_MIN; i <= K_MAX; i *= 2) {
    K = i;
    log_K = (int)log2(K);

    elapsed1 = 0.0;
    elapsed2 = 0.0;
    elapsed3 = 0.0;
    elapsed4 = 0.0;

    for (iter = 0; iter < TRIALS; iter++) {
      u = (double)rand()/(double)(RAND_MAX);

      // reading the relative probabilities array of K elements
      double* a = (double*) malloc(K * sizeof(double));
      initArray(a, K, argv[1]);

      // creating GPU memory for the array
      double* dp;
      cudaMalloc(&dp, K * sizeof(double));


      // version 1: using linear search on prefix sum array

      // copying the array into GPU memory
      cudaMemcpy(dp, a, K * sizeof(double), cudaMemcpyHostToDevice);

      // kernel launch - with timing
      cudaEventRecord(start, 0);
      prefix_linear_kernel<<<1,1>>>(dp, u, dj, K);
      cudaEventRecord(end, 0);
      cudaDeviceSynchronize();

      // elapsed time
      cudaEventElapsedTime(&elapsed, start, end);
      elapsed1 += elapsed;


      // version 2: using binary search on prefix sum array

      // copying the array into GPU memory
      cudaMemcpy(dp, a, K * sizeof(double), cudaMemcpyHostToDevice);

      // kernel launch - with timing
      cudaEventRecord(start, 0);
      prefix_binary_kernel<<<1,1>>>(dp, u, dj, K);
      cudaEventRecord(end, 0);
      cudaDeviceSynchronize();

      // elapsed time
      cudaEventElapsedTime(&elapsed, start, end);
      elapsed2 += elapsed;


      // version 3: using modified binary search on partial sum array

      // copying the array into GPU memory
      cudaMemcpy(dp, a, K * sizeof(double), cudaMemcpyHostToDevice);

      // kernel launch - with timing
      cudaEventRecord(start, 0);
      partial_binary_kernel<<<1,1>>>(dp, u, dj, K, log_K);
      cudaEventRecord(end, 0);
      cudaDeviceSynchronize();

      // elapsed time
      cudaEventElapsedTime(&elapsed, start, end);
      elapsed3 += elapsed;


      // version 4: using parallel search on prefix sum array computed in parallel

      // copying the array into GPU memory
      cudaMemcpy(dp, a, K * sizeof(double), cudaMemcpyHostToDevice);

      // initialising the sampled index to 0 - since it will be atomically updated later
      cudaMemset(&dj, 0, 1);

      // kernel launches - with timing
      cudaEventRecord(start, 0);
      prefix_sum_kernel<<<1, K>>>(dp, K);
      sample_element_kernel<<<1, K>>>(dp, dj, K, u);
      cudaEventRecord(end, 0);
      cudaDeviceSynchronize();

      // elapsed time
      cudaEventElapsedTime(&elapsed, start, end);
      elapsed4 += elapsed;
    }

    printf("%d,%.4f,%.4f,%.4f,%.4f\n", K, elapsed1/TRIALS, elapsed2/TRIALS, elapsed3/TRIALS, elapsed4/TRIALS);
  }

  cudaEventDestroy(start);
  cudaEventDestroy(end);

  return 0;
}
