// topic sampling using butterfly-patterned partial sums for an LDA application

// compilation and execution commands:
// nvcc -arch=sm_35 -rdc=true lda_sampling_c.cu
// ./a.out lda_toi.txt >> lda_sampling_c_outputs.txt

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

#define TRIALS 100
#define K 240
#define BLOCKSIZE 1024
#define W 32
#define LOG2W 5

__global__ void init_theta_kernel(int M, double* dtheta, unsigned int seed) {
  // the document number to be processed by this thread
  int uid = blockIdx.x * blockDim.x + threadIdx.x;

  if (uid < M) {
    // random float number generaion
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

__global__ void init_phi_kernel(int V, double* dphi, unsigned int seed) {
  // te=he topic number to be processed by this thread
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

__global__ void lda_butterfly_kernel(int M, int V, int* dN, int* dw, int* dwind, double* dphi, double* dtheta, int* dz, double u) {
  // the warp number
  int q = blockIdx.x;

  // the lane number within this warp
  int r = threadIdx.x;

  // the document to be processed by this thread
  int m = q * W + r;

  int i, i1, i_master, j, k, c, mid, b, bit, d, searchBase, blockBase, hisBlockBase, flip, mask, him;
  double u1, sum, lowValue, highValue, compareValue, h, v, t, y;

  // local arrays for sampling purpose
  double dtheta_local[K];
  double dp_local[K];

  // ideally these local arrays will be stored in the thread-local registers
  double da_local_reg[W];
  int dc_warp_reg[W];

  // caching theta values (transposed access)
  j = 0;

  // for remnants
  while (j < K%W) {
    dtheta_local[j] = dtheta[m * K + j];
    j++;
  }

  // for blocks of size W*W
  while (j < K) {
    for (k = 0; k < W; k++) {
      dtheta_local[j+k] = dtheta[(q * W + k) * K + j + r];
    }

    j += W;
  }

  i_master = 0;

  // warp voting in action - note that m is a function of threadIdx.x
  while (__any_sync(0xffffffff, (i_master < dN[m]))) {
    // i is minimum of dN[m]-1, i_master
    i = (i_master < dN[m] - 1) ? i_master : (dN[m] - 1);

    // computing butterfly-patterned table of partial sums
    c = dw[dwind[m] + i];
    for (k = 0; k < W; k++) {
      dc_warp_reg[k] = __shfl_sync(0xffffffff, c, k, W);
    }

    sum = 0.0;
    j = 0;

    // for the remnant
    while (j < K%W) {
      sum += (dtheta_local[j] * dphi[c * K + j]);
      dp_local[j] = sum;
      j++;
    }

    // for the W*W blocks
    while (j < K) {
      // transposed access to phi array
      for (k = 0; k < W; k++) {
        da_local_reg[k] = dtheta_local[j + k] * dphi[dc_warp_reg[k] * K + j + r];
      }
      for (b = 0; b < LOG2W; b++) {
        bit = (int)pow(2, b);
        for (i1 = 0; i1 < (W/(2*bit)); i1++) {
          d = 2 * bit * i1 + (bit - 1);
          h = (m & bit) ? da_local_reg[d] : da_local_reg[d + bit];
          v = __shfl_xor_sync(0xffffffff, h, bit, W);
          if (r & bit) {
            da_local_reg[d] = da_local_reg[d + bit];
          }
          da_local_reg[d + bit] = da_local_reg[d] + v;
          dp_local[j + d] = da_local_reg[d];
        }
      }
      sum += da_local_reg[W - 1];
      dp_local[W - 1] = sum;
      j += W;
    }

    // searching within the butterfly-patterned table of partial sums
    u1 = sum * u;
    j = 0;
    k = (K/W) - 1;
    searchBase = (K%W) + W - 1;
    while (j < k) {
      mid = (j+k)/2;
      if (u1 < dp_local[mid*W + searchBase]) {
        k = mid;
      }
      else {
        j = mid+1;
      }
    }

    blockBase = (K%W) + j * W;
    if (K >= W) {
      // butterfly search within one W*W block
      lowValue = (blockBase > 0) ? dp_local[blockBase - 1] : 0;
      highValue = dp_local[blockBase + W - 1];
      flip = 0;

      // butterfly search within block of size W
      for (b = 0; b < LOG2W; b++) {
        bit = (int)pow(2, LOG2W - 1 - b);
        mask = ((W - 1) * 2 * bit) & (W - 1);
        y = 0;
        for (i1 = 0; i1 < W/(2 * bit); i1++) {
          d = (bit - 1) + (2 * bit * i1);
          him = (d & mask) + (r & (~mask));
          hisBlockBase = __shfl_sync(0xffffffff, blockBase, him, W);
          t = __shfl_xor_sync(0xffffffff, dp_local[hisBlockBase + d], flip, W);
          if ((r ^ d) & mask == 0) {
            y = t;
          }
        }
        compareValue = (r & bit) ? highValue - y : lowValue + y;
        if (u1  < compareValue) {
          highValue = compareValue;
          flip = flip ^ (bit & r);
        }
        else {
          lowValue = compareValue;
          flip = flip ^ (bit & ~r);
        }
      }

      j = blockBase + (flip ^ r);
    }

    if (blockBase > 0) {
      if (u1 < dp_local[blockBase - 1]) {
        for (i1 = 0; i1 < K%W; i1++) {
          if (u1 < dp_local[i1]) {
            j = i1;
            break;
          }
        }
      }
    }

    // storing the sampled topic for this word
    dz[dwind[m] + i] = j;
    i_master++;
  }
}

__global__ void reinit_theta_kernel(int M, double* dtheta) {
  int uid = blockIdx.x * blockDim.x + threadIdx.x;

  // re-initialises the theta entries to zero
  if (uid < M * K) {
    dtheta[uid] = 0.0;
  }
}

__global__ void reinit_phi_kernel(int V, double* dphi) {
  int uid = blockIdx.x * blockDim.x + threadIdx.x;

  // re-initialises the phi entries to zero
  if (uid < V * K) {
    dphi[uid] = 0.0;
  }
}

__global__ void recalculate_params_from_topics_kernel(int M, int* dN, int* dw, int* dwind, double* dphi, double* dtheta, int* dz) {
  // the document to be processed by this thread
  int uid = blockIdx.x * blockDim.x + threadIdx.x;

  if (uid < M) {
    int i, w, c, t;

    // going through all the words of this document
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

__global__ void update_theta_kernel(int M, int* dN, double* dtheta) {
  // the document number to be processed by this thread
  int uid = blockIdx.x * blockDim.x + threadIdx.x;

  if (uid < M) {
    // normalising the entries of one row of the theta array (since it forms a discrete probability distribution)
    int i;
    for (i = 0; i < K; i++) {
      dtheta[uid * K + i] /= dN[uid];
    }
  }
}

__global__ void update_phi_kernel(int V, double* dphi) {
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

__global__ void normalize_theta_kernel(int M, double* dtheta) {
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

  // to be used in random sampling
  double u;

  // to time the execution of kernels
  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  float elapsed;

  // number of thread blocks that the kernels will be launched with
  int numBlocks = ceil((float)M/BLOCKSIZE);

  // number of warps which will be in action; assumes that M is a multiple of W
  int numWarps = M/W;

  int iter;

  int numBlocks1 = ceil((float)(M * K)/BLOCKSIZE);
  int numBlocks2 = ceil((float)(V * K)/BLOCKSIZE);
  int numBlocks3 = ceil((float)(totWords * K)/BLOCKSIZE);

  // creating and initialising GPU memory for theta array (M * K), which represents document-wise distribution of topics
  double* dtheta;
  cudaMalloc(&dtheta, M * K * sizeof(double));
  init_theta_kernel<<<numBlocks, BLOCKSIZE>>>(M, dtheta, time(NULL));
  cudaDeviceSynchronize();

  // creating and initialising GPU memory for phi array (V * K), which represents topic-wise distribution of words
  double* dphi;
  cudaMalloc(&dphi, V * K * sizeof(double));
  init_phi_kernel<<<1, K>>>(V, dphi, time(NULL)); // since K can at most be 256, one thread block will be sufficient
  cudaDeviceSynchronize();


  // timing the execution of kernels required for LDA topic-sampling

  cudaEventRecord(start, 0);

  for (iter = 0; iter < TRIALS; iter++) {
    u = (double)rand()/(double)(RAND_MAX);
    lda_butterfly_kernel<<<numWarps, W>>>(M, V, dN, dw, dwind, dphi, dtheta, dz, u);
    reinit_theta_kernel<<<numBlocks1, BLOCKSIZE>>>(M, dtheta);
    reinit_phi_kernel<<<numBlocks2, BLOCKSIZE>>>(V, dphi);
    recalculate_params_from_topics_kernel<<<numBlocks, BLOCKSIZE>>>(M, dN, dw, dwind, dphi, dtheta, dz);
    update_theta_kernel<<<numBlocks, BLOCKSIZE>>>(M, dN, dtheta);
    update_phi_kernel<<<1, K>>>(V, dphi);
  }

  normalize_theta_kernel<<<numBlocks, BLOCKSIZE>>>(M, dtheta);

  cudaEventRecord(end, 0);

  cudaDeviceSynchronize();

  cudaEventElapsedTime(&elapsed, start, end);
  printf("%d,%.4f\n", K, elapsed/TRIALS);

  cudaEventDestroy(start);
  cudaEventDestroy(end);

  return 0;
}
