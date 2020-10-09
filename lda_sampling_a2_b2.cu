// experiments with different versions of topic sampling for an LDA application

// compilation and execution commands:
// nvcc -arch=sm_35 -rdc=true lda_sampling_a2_b2.cu
// ./a.out lda_toi.txt > lda_sampling_a2_b2_outputs.txt

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

#define TRIALS 1
#define K_MIN 16
#define K_MAX 240
#define BLOCKSIZE 1024

__global__ void init_theta_kernel(int M, int K, double* dtheta, unsigned int seed) {
  // the document number to be processed by this thread
  int uid = blockIdx.x * blockDim.x + threadIdx.x;

  if (uid < M) {
    // random float number generation
    curandState_t state;
    curand_init(seed, 0, 0, &state);

    // populating one row of the theta array
    double sum = 0.0;
    double num;
    int j;
    for (j = 0; j < K; j++) {
      num = (float)(curand(&state)%100)/100.0;
      dtheta[uid * K + j] = num;
      sum += num;
    }

    // normalising the entries since each row is a discrete probability distribution
    for (j = 0; j < K; j++) {
      dtheta[uid * K + j] /= sum;
    }
  }
}

__global__ void init_phi_kernel(int V, int K, double* dphi, unsigned int seed) {
  // the topic number to be processed by this thread
  int uid = threadIdx.x;

  // random float number generation
  curandState_t state;
  curand_init(seed, 0, 0, &state);

  // populating one column of the phi array
  double sum = 0.0;
  double num;
  int i;
  for (i = 0; i < V; i++) {
    (float)(curand(&state)%100)/100.0;
    dphi[i * K + uid] = num;
    sum += num;
  }

  // normalising the entries since each column is a discrete probability distribution
  for (i = 0; i < V; i++) {
    dphi[i * K + uid] /= sum;
  }
}

__global__ void prefix_sum_kernel(double* da, int wn, int K) {
  int uid = threadIdx.x;

  // one step in the parallel computation of prefix sums
  int off;
  double temp;

  for (off = 1; off < K; off *= 2) {
    // first we perform the reads
    if (threadIdx.x >= off) {
        temp = da[wn * K + uid - off];
    }
    __syncthreads();

    // then we perform the writes
    if (threadIdx.x >= off) {
        da[wn * K + uid] += temp;
    }
    __syncthreads();
  }
}

__global__ void lda_prefix_linear_kernel(int M, int K, int* dN, int* dw, int* dwind, double* dphi, double* dtheta, double* da, int* dz, double u) {
  // the document number ot be processed by this thread
  int uid = blockDim.x * blockIdx.x + threadIdx.x;
  if (uid < M) {
    int i, j;

    // sampling a topic for each word in the document
    for (i = 0; i < dN[uid]; i++) {
      // computing the array of relative probabilities to sample a topic for this word
      int wn = dwind[uid] + i;
      for (j = 0; j < K; j++) {
        da[wn * K + j] = dtheta[uid * K + j] * dphi[dw[wn] * K + j];
      }

      // computing the prefix sums
      prefix_sum_kernel<<<1, K>>>(da, wn, K);
      cudaDeviceSynchronize();

      // cumulative probability to be searched for
      double u1 = u * da[wn * K + K - 1];

      // linear search on prefix sum array
      j = 0;
      while (j < K-1 && u1 >= da[wn * K + j]) {
        j++;
      }

      // storing the sampled topic
      dz[wn] = j;
    }
  }
}

__global__ void lda_prefix_binary_kernel(int M, int K, int* dN, int* dw, int* dwind, double* dphi, double* dtheta, double* da, int* dz, double u) {
  // the document number to be processed by this thread
  int uid = blockDim.x * blockIdx.x + threadIdx.x;

  if (uid < M) {
    int i, j, k, mid;

    // sampling a topic for each word in the document
    for (i = 0; i < dN[uid]; i++) {
      // computing the array of relative probabilities to sample a topic for this word
      int wn = dwind[uid] + i;
      for (j = 0; j < K; j++) {
        da[wn * K + j] = dtheta[uid * K + j] * dphi[dw[wn] * K + j];
      }

      // computing the prefix sum
      prefix_sum_kernel<<<1, K>>>(da, wn, K);
      cudaDeviceSynchronize();

      // cumulative probability to be searched for
      double u1 = u * da[wn * K + K - 1];

      // binary search on prefix sum array
      j = 0;
      k = K - 1;
      while (j < k) {
        mid = (j + k)/2;
        if (u1 < da[wn * K + mid]) {
          k = mid;
        }
        else {
          j = mid + 1;
        }
      }

      // storing the sampled index
      dz[wn] = j;
    }
  }
}

__global__ void reinit_theta_kernel(int M, int K, double* dtheta) {
  int uid = blockIdx.x * blockDim.x + threadIdx.x;

  // re-initialises the theta entries to zero
  if (uid < M * K) {
    dtheta[uid] = 0.0;
  }
}

__global__ void reinit_phi_kernel(int V, int K, double* dphi) {
  int uid = blockIdx.x * blockDim.x + threadIdx.x;

  // re-initialises the phi entries to zero
  if (uid < V * K) {
    dphi[uid] = 0.0;
  }
}

__global__ void recalculate_params_from_topics_kernel(int M, int K, int* dN, int* dw, int* dwind, double* dphi, double* dtheta, int* dz) {
  // the document to be processed by this thread
  int uid = blockIdx.x * blockDim.x + threadIdx.x;

  if (uid < M) {
    int i, w, c, t;

    // going through all the words of the document
    for (i = 0; i < dN[uid]; i++) {
      // the index of this word into the word ids array
      w = dwind[uid] + i;

      // the unique id of this word
      c = dw[w];

      // the topic assigned to this word
      t = dz[w];

      // updating the counts in the theta and phi arrays accordingly
      dtheta[uid * K + t] = dtheta[uid * K + t] + 1.0;
      dphi[c * K + t] = dphi[c * K + t] + 1.0;
    }
  }
}

__global__ void update_theta_kernel(int M, int K, int* dN, double* dtheta) {
  // the document to be processed by this thread
  int uid = blockIdx.x * blockDim.x + threadIdx.x;

  if (uid < M) {
    // normalising the entries of one row of the theta array (since it forms a discrete probability distribution)
    int i;
    for (i = 0; i < K; i++) {
      dtheta[uid * K + i] /= dN[uid];
    }
  }
}

__global__ void update_phi_kernel(int V, int K, double* dphi) {
  // the topic number to be processed by this thread
  int uid = blockIdx.x * blockDim.x + threadIdx.x;

  // normalising the entries of one column of the phi array (since it forms a discrete probability distribution)
  int i;
  double sum = 0.0;
  for (i = 0; i < V; i++) {
    sum += dphi[i * K + uid];
  }

  // sanity check to avoid division by zero
  if (sum > 0.0) {
    for (i = 0; i < V; i++) {
      dphi[i * K + uid] /= sum;
    }
  }
}

__global__ void normalize_theta_kernel(int M, int K, double* dtheta) {
  // the document to be processed by this thread
  int uid = blockIdx.x * blockDim.x + threadIdx.x;

  if (uid < M) {
    // normalising the entries of one row of the theta array (since it forms a discrete probability distribution)
    int i;
    double sum = 0.0;

    for (i = 0; i < K; i++) {
      sum += dtheta[uid * K + i];
    }

    for (i = 0; i < K; i++) {
      dtheta[uid * K + i] /= sum;
    }
  }
}

// takes as command-line argument the path to the input file
int main(int argc, char* argv[]) {
  // number of documents
  int M;

  // size of vocabulary
  int V;

  // number of elements to be sampled from
  int K;

  FILE *f = fopen(argv[1], "r");
  fscanf(f, "%d", &M);
  fscanf(f, "%d", &V);

  // number of words in each document
  int* N = (int*) malloc(M * sizeof(int));

  // starting indices of the words of each document in the word array
  int* wind = (int*) malloc(M * sizeof(int));

  int i;

  // total number of words in all documents
  int totWords = 0;

  // reading the number of words in each document and populating the starting word indices
  wind[0] = 0;
  for (i = 0; i < M-1; i++) {
      fscanf(f, "%d", &N[i]);
      totWords += N[i];
      wind[i+1] = totWords;
  }
  fscanf(f, "%d", &N[M-1]);
  totWords += N[M-1];

  // creating and initialising GPU memory for number of words in each document
  int* dN;
  cudaMalloc(&dN, M * sizeof(int));
  cudaMemcpy(dN, N, M * sizeof(int), cudaMemcpyHostToDevice);

  // reading the word numbers for the document-wise list of words
  int* w = (int*) malloc(totWords * sizeof(int));
  for (i = 0; i < totWords; i++) {
      fscanf(f, "%d", &w[i]);
  }

  // creating and initialising GPU memory for the word numbers of document-wise list of words
  int* dw;
  cudaMalloc(&dw, totWords * sizeof(int));
  cudaMemcpy(dw, w, totWords * sizeof(int), cudaMemcpyHostToDevice);

  // creating and initialising GPU memory for the starting word indices
  int* dwind;
  cudaMalloc(&dwind, M * sizeof(int));
  cudaMemcpy(dwind, wind, M * sizeof(int), cudaMemcpyHostToDevice);

  fclose(f);

  // to store the sampled topic for each document
  int* dz;
  cudaMalloc(&dz, totWords * sizeof(int));
  cudaMemset(&dz, 0, totWords * sizeof(int));

  // to be used in random sampling
  double u;

  // to time the execution of kernels
  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  float elapsed, elapsed1, elapsed2;

  // number of thread blocks that the kernels will be launched with
  int numBlocks = ceil((float)M/BLOCKSIZE);

  int iter;
  printf("K,t1,t2\n");
  for (i = K_MIN; i <= K_MAX; i += 32) {
    K = i;
    int numBlocks1 = ceil((float)(M * K)/BLOCKSIZE);
    int numBlocks2 = ceil((float)(V * K)/BLOCKSIZE);

    elapsed1 = 0.0;
    elapsed2 = 0.0;

    // creating and initialising GPU memory for theta array (M * K), which represents document-wise distribution of topics
    double* dtheta;
    cudaMalloc(&dtheta, M * K * sizeof(double));
    init_theta_kernel<<<numBlocks, BLOCKSIZE>>>(M, K, dtheta, time(NULL));
    cudaDeviceSynchronize();

    // creating and initialising GPU memory for phi array (V * K), which represents topic-wise distribution of words
    double* dphi;
    cudaMalloc(&dphi, V * K * sizeof(double));
    init_phi_kernel<<<1, K>>>(V, K, dphi, time(NULL)); // since K can at most be 256, one thread block will be sufficient
    cudaDeviceSynchronize();

    // to store the element wise product of theta and phi arrays for each word in each document
    // this gives the relative probability array to be sampled from
    // this will be reused to store the prefix/partial sum arrays as well
    double* da;
    cudaMalloc(&da, totWords * K * sizeof(double));

    // version 1: using linear search on prefix sum array

    cudaEventRecord(start, 0);

    for (iter = 0; iter < TRIALS; iter++) {
      u = (double)rand()/(double)(RAND_MAX);
      lda_prefix_linear_kernel<<<numBlocks, BLOCKSIZE>>>(M, K, dN, dw, dwind, dphi, dtheta, da, dz, u);
      reinit_theta_kernel<<<numBlocks1, BLOCKSIZE>>>(M, K, dtheta);
      reinit_phi_kernel<<<numBlocks2, BLOCKSIZE>>>(V, K, dphi);
      recalculate_params_from_topics_kernel<<<numBlocks, BLOCKSIZE>>>(M, K, dN, dw, dwind, dphi, dtheta, dz);
      update_theta_kernel<<<numBlocks, BLOCKSIZE>>>(M, K, dN, dtheta);
      update_phi_kernel<<<1, K>>>(V, K, dphi);
    }

    normalize_theta_kernel<<<numBlocks, BLOCKSIZE>>>(M, K, dtheta);

    cudaEventRecord(end, 0);

    cudaDeviceSynchronize();

    cudaEventElapsedTime(&elapsed, start, end);
    elapsed1 = elapsed/TRIALS;


    // version 2: using binary search on prefix sum array

    cudaEventRecord(start, 0);

    for (iter = 0; iter < TRIALS; iter++) {
      u = (double)rand()/(double)(RAND_MAX);
      lda_prefix_binary_kernel<<<numBlocks, BLOCKSIZE>>>(M, K, dN, dw, dwind, dphi, dtheta, da, dz, u);
      reinit_theta_kernel<<<numBlocks1, BLOCKSIZE>>>(M, K, dtheta);
      reinit_phi_kernel<<<numBlocks2, BLOCKSIZE>>>(V, K, dphi);
      recalculate_params_from_topics_kernel<<<numBlocks, BLOCKSIZE>>>(M, K, dN, dw, dwind, dphi, dtheta, dz);
      update_theta_kernel<<<numBlocks, BLOCKSIZE>>>(M, K, dN, dtheta);
      update_phi_kernel<<<1, K>>>(V, K, dphi);
    }

    normalize_theta_kernel<<<numBlocks, BLOCKSIZE>>>(M, K, dtheta);

    cudaEventRecord(end, 0);

    cudaDeviceSynchronize();

    cudaEventElapsedTime(&elapsed, start, end);
    elapsed2 = elapsed/TRIALS;

    printf("%d,%.4f,%.4f\n", K, elapsed1, elapsed2);
  }

  cudaEventDestroy(start);
  cudaEventDestroy(end);

  return 0;
}
