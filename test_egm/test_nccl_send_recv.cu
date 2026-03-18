/*
 * NCCL Send/Recv: cross-node GPU→CPU benchmark
 *
 * 4 nodes, 1 rank per node (GPU0), ranks: R0=Node0 R1=Node1 R2=Node2 R3=Node3
 *
 *   Case 1: Node{0,1} GPU0 HBM send → Node{2,3} recv varies  (R0→R2, R1→R3)
 *   Case 2: Node{2,3} send varies   → Node{0,1} GPU0 HBM recv (R2→R0, R3→R1)
 *
 * Buffer types (the "varies" side cycles through):
 *   A) HBM:    cudaMalloc (GPU device memory)
 *   B) Pinned: mmap+mbind(NUMA0)+cudaHostRegister (NUMA-aware pinned host)
 *   C) EGM:    cuMemCreate(HOST_NUMA 0) + cuMemMap (GPU-accessible host)
 *
 * Build:
 *   nvcc -o test_nccl_send_recv test_nccl_send_recv.cu \
 *        -lnccl -lmpi -lcuda -lnuma -std=c++17 -arch=native
 *
 * Run:
 *   srun -N4 --ntasks-per-node=1 --gpus-per-node=1 ./test_nccl_send_recv
 */

#include <mpi.h>
#include <nccl.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <numaif.h>
#include <numa.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cfloat>

static int g_rank = 0;

#define MPI_CHECK(cmd) do {                                     \
  int _e = (cmd);                                               \
  if (_e != MPI_SUCCESS) {                                      \
    char estr[MPI_MAX_ERROR_STRING]; int elen;                  \
    MPI_Error_string(_e, estr, &elen);                          \
    fprintf(stderr, "R%d MPI error %s:%d: %s\n",               \
            g_rank, __FILE__, __LINE__, estr);                  \
    MPI_Abort(MPI_COMM_WORLD, 1);                               \
  }                                                             \
} while(0)

#define NCCL_CHECK(cmd) do {                                    \
  ncclResult_t _e = (cmd);                                      \
  if (_e != ncclSuccess) {                                      \
    fprintf(stderr, "R%d NCCL error %s:%d: %s\n",              \
            g_rank, __FILE__, __LINE__, ncclGetErrorString(_e));\
    MPI_Abort(MPI_COMM_WORLD, 1);                               \
  }                                                             \
} while(0)

#define CUDA_CHECK(cmd) do {                                    \
  cudaError_t _e = (cmd);                                       \
  if (_e != cudaSuccess) {                                      \
    fprintf(stderr, "R%d CUDA error %s:%d: %s\n",              \
            g_rank, __FILE__, __LINE__, cudaGetErrorString(_e));\
    MPI_Abort(MPI_COMM_WORLD, 1);                               \
  }                                                             \
} while(0)

#define CU_CHECK(cmd) do {                                      \
  CUresult _e = (cmd);                                          \
  if (_e != CUDA_SUCCESS) {                                     \
    const char *_s = nullptr; cuGetErrorString(_e, &_s);        \
    fprintf(stderr, "R%d CU error %s:%d: %s (%d)\n",           \
            g_rank, __FILE__, __LINE__, _s?_s:"?", (int)_e);   \
    MPI_Abort(MPI_COMM_WORLD, 1);                               \
  }                                                             \
} while(0)

// ==================== EGM (cuMem HOST_NUMA) ====================

struct EgmBuffer {
  CUdeviceptr devPtr;
  CUmemGenericAllocationHandle handle;
  size_t allocSize;
};

static bool egm_alloc(EgmBuffer *buf, size_t size, int cudaDev, int numaId) {
  CUdevice device;
  CU_CHECK(cuDeviceGet(&device, cudaDev));

  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
  prop.location.id = numaId;

  size_t granularity = 0;
  CU_CHECK(cuMemGetAllocationGranularity(
      &granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));

  size_t aligned = ((size + granularity - 1) / granularity) * granularity;

  CUresult res = cuMemCreate(&buf->handle, aligned, &prop, 0);
  if (res != CUDA_SUCCESS) {
    const char *s = nullptr; cuGetErrorString(res, &s);
    fprintf(stderr, "R%d: cuMemCreate NUMA %d size %zu failed: %s\n",
            g_rank, numaId, aligned, s ? s : "?");
    return false;
  }

  CUdeviceptr dptr = 0;
  CU_CHECK(cuMemAddressReserve(&dptr, aligned, granularity, 0, 0));
  CU_CHECK(cuMemMap(dptr, aligned, 0, buf->handle, 0));

  CUmemAccessDesc acc = {};
  acc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  acc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  acc.location.id = cudaDev;
  CU_CHECK(cuMemSetAccess(dptr, aligned, &acc, 1));

  acc.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
  acc.location.id = numaId;
  CU_CHECK(cuMemSetAccess(dptr, aligned, &acc, 1));

  buf->devPtr = dptr;
  buf->allocSize = aligned;
  return true;
}

static void egm_free(EgmBuffer *buf) {
  if (buf->devPtr) {
    cuMemUnmap(buf->devPtr, buf->allocSize);
    cuMemRelease(buf->handle);
    cuMemAddressFree(buf->devPtr, buf->allocSize);
    buf->devPtr = 0;
  }
}

// ==================== Pinned Host on NUMA ====================

struct PinnedBuffer {
  void *ptr;
  size_t allocSize;
};

static bool pinned_alloc_on_numa(PinnedBuffer *buf, size_t size, int numaId) {
  long page_size = sysconf(_SC_PAGESIZE);
  size_t aligned = ((size + page_size - 1) / page_size) * page_size;

  void *ptr = mmap(NULL, aligned, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (ptr == MAP_FAILED) return false;

  unsigned long nodemask = 1UL << numaId;
  if (mbind(ptr, aligned, MPOL_BIND, &nodemask,
            sizeof(nodemask) * 8, MPOL_MF_MOVE | MPOL_MF_STRICT) != 0) {
    munmap(ptr, aligned);
    return false;
  }
  memset(ptr, 0, aligned);

  cudaError_t err = cudaHostRegister(ptr, aligned, cudaHostRegisterDefault);
  if (err != cudaSuccess) {
    munmap(ptr, aligned);
    return false;
  }

  buf->ptr = ptr;
  buf->allocSize = aligned;
  return true;
}

static void pinned_free(PinnedBuffer *buf) {
  if (buf->ptr) {
    cudaHostUnregister(buf->ptr);
    munmap(buf->ptr, buf->allocSize);
    buf->ptr = nullptr;
  }
}

// ==================== GPU fill / verify kernels ====================

__global__ void fill_pattern(float *ptr, size_t n, float val) {
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) ptr[idx] = val + (float)(idx & 0xFF);
}

__global__ void verify_pattern(const float *ptr, size_t n, float expected_base,
                               int *error_count, size_t *first_err_idx,
                               float *first_got, float *first_exp) {
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    float exp = expected_base + (float)(idx & 0xFF);
    if (ptr[idx] != exp) {
      int old = atomicAdd(error_count, 1);
      if (old == 0) {
        *first_err_idx = idx;
        *first_got = ptr[idx];
        *first_exp = exp;
      }
    }
  }
}

// ==================== Benchmark ====================

struct BenchResult { double bw_gb_s; double avg_us; bool verified; };

/*
 * Paired send/recv between (sender0→recver0) and (sender1→recver1).
 * All 4 ranks participate in the ncclGroup so NCCL can progress;
 * non-involved ranks simply idle inside the group.
 *
 * sender_ranks[2]: the two ranks that send
 * recver_ranks[2]: the two ranks that recv
 * For a given rank, if it's a sender it uses sendbuf to send to its paired recver;
 * if it's a recver it uses recvbuf to receive from its paired sender.
 */
static BenchResult run_paired_bench(
    void *sendbuf, void *recvbuf, size_t count,
    ncclComm_t comm, cudaStream_t stream,
    int my_rank,
    const int sender_ranks[2], const int recver_ranks[2],
    int warmup, int iters)
{
  int my_peer = -1;
  bool i_send = false, i_recv = false;
  for (int i = 0; i < 2; i++) {
    if (my_rank == sender_ranks[i]) { my_peer = recver_ranks[i]; i_send = true; }
    if (my_rank == recver_ranks[i]) { my_peer = sender_ranks[i]; i_recv = true; }
  }

  for (int w = 0; w < warmup; w++) {
    NCCL_CHECK(ncclGroupStart());
    if (i_send) NCCL_CHECK(ncclSend(sendbuf, count, ncclFloat, my_peer, comm, stream));
    if (i_recv) NCCL_CHECK(ncclRecv(recvbuf, count, ncclFloat, my_peer, comm, stream));
    NCCL_CHECK(ncclGroupEnd());
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));

  cudaEvent_t t0, t1;
  CUDA_CHECK(cudaEventCreate(&t0));
  CUDA_CHECK(cudaEventCreate(&t1));

  CUDA_CHECK(cudaEventRecord(t0, stream));
  for (int i = 0; i < iters; i++) {
    NCCL_CHECK(ncclGroupStart());
    if (i_send) NCCL_CHECK(ncclSend(sendbuf, count, ncclFloat, my_peer, comm, stream));
    if (i_recv) NCCL_CHECK(ncclRecv(recvbuf, count, ncclFloat, my_peer, comm, stream));
    NCCL_CHECK(ncclGroupEnd());
  }
  CUDA_CHECK(cudaEventRecord(t1, stream));
  CUDA_CHECK(cudaEventSynchronize(t1));

  float ms = 0;
  CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
  cudaEventDestroy(t0);
  cudaEventDestroy(t1);

  double bytes = (double)count * sizeof(float);
  double avg_us = (ms * 1000.0) / iters;
  double bw = (i_send || i_recv) && avg_us > 0
              ? bytes / (avg_us / 1e6) / 1e9 : 0;

  // Verify: recv buffer should contain sender's pattern
  bool verified = true;
  if (i_recv && recvbuf) {
    float expected_base = (float)(my_peer + 1) * 1000.0f;

    int *d_err_count;
    size_t *d_first_idx;
    float *d_first_got, *d_first_exp;
    CUDA_CHECK(cudaMalloc(&d_err_count, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_first_idx, sizeof(size_t)));
    CUDA_CHECK(cudaMalloc(&d_first_got, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_first_exp, sizeof(float)));
    CUDA_CHECK(cudaMemsetAsync(d_err_count, 0, sizeof(int), stream));

    size_t grid = (count + 255) / 256;
    verify_pattern<<<(int)grid, 256, 0, stream>>>(
        (const float*)recvbuf, count, expected_base,
        d_err_count, d_first_idx, d_first_got, d_first_exp);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    int h_err_count = 0;
    CUDA_CHECK(cudaMemcpy(&h_err_count, d_err_count, sizeof(int), cudaMemcpyDeviceToHost));

    if (h_err_count > 0) {
      verified = false;
      size_t h_idx; float h_got, h_exp;
      CUDA_CHECK(cudaMemcpy(&h_idx, d_first_idx, sizeof(size_t), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(&h_got, d_first_got, sizeof(float), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(&h_exp, d_first_exp, sizeof(float), cudaMemcpyDeviceToHost));
      fprintf(stderr, "  R%d VERIFY FAIL: %d errors, first at idx %zu: got %f, expected %f (src R%d)\n",
              g_rank, h_err_count, h_idx, h_got, h_exp, my_peer);
    }

    cudaFree(d_err_count);
    cudaFree(d_first_idx);
    cudaFree(d_first_got);
    cudaFree(d_first_exp);
  }

  return {bw, avg_us, verified};
}

// ==================== Run one case with 3 memory types ====================
//
// vary_send=false: send buf is always GPU HBM, recv buf varies (HBM/Pinned/EGM)
// vary_send=true:  send buf varies (HBM/Pinned/EGM), recv buf is always GPU HBM

static constexpr int NUM_MEM_TYPES = 3;

static void run_case(
    const char *case_name,
    const int sender_ranks[2], const int recver_ranks[2],
    int my_rank, int gpu_id, int numa_id, bool vary_send,
    ncclComm_t comm, cudaStream_t stream)
{
  bool i_send = (my_rank == sender_ranks[0] || my_rank == sender_ranks[1]);
  bool i_recv = (my_rank == recver_ranks[0] || my_rank == recver_ranks[1]);
  bool i_vary = vary_send ? i_send : i_recv;

  struct SizeEntry { size_t elems; const char *label; };
  SizeEntry sizes[] = {
    {    64*1024*1024ULL,   " 256 MB"},
    {   256*1024*1024ULL,   "   1 GB"},
    {   512*1024*1024ULL,   "   2 GB"},
    {  4096*1024*1024ULL,   "  16 GB"},
  };
  int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
  int warmup = 5;
  int iters  = 20;

  if (my_rank == 0) {
    printf("\n  ============ %s ============\n", case_name);
    printf("  Senders: R%d(Node%d), R%d(Node%d)  -->  Receivers: R%d(Node%d), R%d(Node%d)\n",
           sender_ranks[0], sender_ranks[0],
           sender_ranks[1], sender_ranks[1],
           recver_ranks[0], recver_ranks[0],
           recver_ranks[1], recver_ranks[1]);
    if (vary_send)
      printf("  Send buf varies (NUMA %d)    Recv buf: GPU HBM (cudaMalloc)\n", numa_id);
    else
      printf("  Send buf: GPU HBM (cudaMalloc)    Recv buf varies (NUMA %d)\n", numa_id);
    printf("  warmup=%d, iter=%d\n\n", warmup, iters);
    printf("  +---------+--------------------+--------------------+--------------------+\n");
    printf("  | Size    | A) HBM             | B) Pinned          | C) EGM             |\n");
    printf("  |         | GB/s    avg_us  chk| GB/s    avg_us  chk| GB/s    avg_us  chk|\n");
    printf("  +---------+--------------------+--------------------+--------------------+\n");
    fflush(stdout);
  }

  for (int s = 0; s < num_sizes; s++) {
    size_t count = sizes[s].elems;
    size_t bytes = count * sizeof(float);

    BenchResult res[NUM_MEM_TYPES] = {};
    bool ok[NUM_MEM_TYPES] = {};

    auto alloc_hbm = [&](float **ptr, bool do_fill) {
      CUDA_CHECK(cudaMalloc(ptr, bytes));
      if (do_fill) {
        size_t grid = (count + 255) / 256;
        fill_pattern<<<(int)grid, 256, 0, stream>>>(*ptr, count, (float)(my_rank+1)*1000.0f);
        CUDA_CHECK(cudaStreamSynchronize(stream));
      }
    };

    // ---------- A) HBM ----------
    {
      float *d_send = nullptr, *d_recv = nullptr;
      if (i_send) alloc_hbm(&d_send, true);
      if (i_recv) alloc_hbm(&d_recv, false);

      res[0] = run_paired_bench(d_send, d_recv, count, comm, stream,
                                my_rank, sender_ranks, recver_ranks, warmup, iters);
      ok[0] = true;

      if (d_send) cudaFree(d_send);
      if (d_recv) cudaFree(d_recv);
    }
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    // ---------- B) Pinned on NUMA ----------
    {
      float *d_fixed = nullptr;
      PinnedBuffer pbuf = {};
      bool pbuf_ok = false;

      if (vary_send) {
        if (i_recv) alloc_hbm(&d_fixed, false);
        if (i_send) {
          pbuf_ok = pinned_alloc_on_numa(&pbuf, bytes, numa_id);
          if (pbuf_ok) {
            size_t grid = (count + 255) / 256;
            fill_pattern<<<(int)grid, 256, 0, stream>>>((float*)pbuf.ptr, count, (float)(my_rank+1)*1000.0f);
            CUDA_CHECK(cudaStreamSynchronize(stream));
          }
        }
      } else {
        if (i_send) alloc_hbm(&d_fixed, true);
        if (i_recv) pbuf_ok = pinned_alloc_on_numa(&pbuf, bytes, numa_id);
      }

      if (i_vary && !pbuf_ok)
        fprintf(stderr, "R%d: pinned_alloc_on_numa(%d) failed for %zu bytes\n",
                my_rank, numa_id, bytes);

      int all_ok_int = (i_vary ? (pbuf_ok ? 1 : 0) : 1);
      int global_ok = 0;
      MPI_Allreduce(&all_ok_int, &global_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

      if (global_ok) {
        void *sendbuf = vary_send ? pbuf.ptr   : (void*)d_fixed;
        void *recvbuf = vary_send ? (void*)d_fixed : pbuf.ptr;
        res[1] = run_paired_bench(sendbuf, recvbuf, count, comm, stream,
                                  my_rank, sender_ranks, recver_ranks, warmup, iters);
        ok[1] = true;
      }

      if (d_fixed) cudaFree(d_fixed);
      if (pbuf_ok) pinned_free(&pbuf);
    }
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    // ---------- C) EGM on NUMA ----------
    {
      float *d_fixed = nullptr;
      EgmBuffer ebuf = {};
      bool ebuf_ok = false;

      if (vary_send) {
        if (i_recv) alloc_hbm(&d_fixed, false);
        if (i_send) {
          ebuf_ok = egm_alloc(&ebuf, bytes, gpu_id, numa_id);
          if (ebuf_ok) {
            size_t grid = (count + 255) / 256;
            fill_pattern<<<(int)grid, 256, 0, stream>>>((float*)ebuf.devPtr, count, (float)(my_rank+1)*1000.0f);
            CUDA_CHECK(cudaStreamSynchronize(stream));
          }
        }
      } else {
        if (i_send) alloc_hbm(&d_fixed, true);
        if (i_recv) ebuf_ok = egm_alloc(&ebuf, bytes, gpu_id, numa_id);
      }

      if (i_vary && !ebuf_ok)
        fprintf(stderr, "R%d: egm_alloc(NUMA %d) failed for %zu bytes\n",
                my_rank, numa_id, bytes);

      int all_ok_int = (i_vary ? (ebuf_ok ? 1 : 0) : 1);
      int global_ok = 0;
      MPI_Allreduce(&all_ok_int, &global_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

      if (global_ok) {
        void *sendbuf = vary_send ? (void*)ebuf.devPtr : (void*)d_fixed;
        void *recvbuf = vary_send ? (void*)d_fixed     : (void*)ebuf.devPtr;
        res[2] = run_paired_bench(sendbuf, recvbuf, count, comm, stream,
                                  my_rank, sender_ranks, recver_ranks, warmup, iters);
        ok[2] = true;
      }

      if (d_fixed) cudaFree(d_fixed);
      if (ebuf_ok) egm_free(&ebuf);
    }
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    // Collect: max BW, min latency, and verify status across all ranks
    double bw_max[NUM_MEM_TYPES], us_min[NUM_MEM_TYPES];
    int verify_ok[NUM_MEM_TYPES];
    for (int i = 0; i < NUM_MEM_TYPES; i++) {
      MPI_Reduce(&res[i].bw_gb_s, &bw_max[i], 1, MPI_DOUBLE, MPI_MAX,
                 0, MPI_COMM_WORLD);
      MPI_Reduce(&res[i].avg_us,  &us_min[i], 1, MPI_DOUBLE, MPI_MIN,
                 0, MPI_COMM_WORLD);
      int my_v = res[i].verified ? 1 : 0;
      MPI_Reduce(&my_v, &verify_ok[i], 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
    }

    if (my_rank == 0) {
      auto fmt = [](bool valid, double bw, double us, int vok) {
        if (!valid) printf("      N/A          ");
        else        printf("%7.2f %9.1f %s", bw, us, vok ? "OK" : "NG");
      };
      printf("  | %s |", sizes[s].label);
      fmt(ok[0], bw_max[0], us_min[0], verify_ok[0]); printf("|");
      fmt(ok[1], bw_max[1], us_min[1], verify_ok[1]); printf("|");
      fmt(ok[2], bw_max[2], us_min[2], verify_ok[2]); printf("|\n");
      fflush(stdout);
    }
  }

  if (my_rank == 0) {
    printf("  +---------+--------------------+--------------------+--------------------+\n");
    fflush(stdout);
  }
  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
}

// ==================== Main ====================

int main(int argc, char **argv) {
  MPI_CHECK(MPI_Init(&argc, &argv));

  int rank, world_size;
  MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &world_size));
  g_rank = rank;

  if (world_size != 4) {
    if (rank == 0)
      fprintf(stderr, "This test requires exactly 4 ranks (1 per node × 4 nodes). Got %d.\n",
              world_size);
    MPI_Finalize();
    return 1;
  }

  // Each rank uses GPU0 on its node
  CUDA_CHECK(cudaSetDevice(0));
  int gpu_id = 0;

  CUdevice cu_dev;
  CU_CHECK(cuDeviceGet(&cu_dev, gpu_id));
  int gpu_numa_id = -1;
  cuDeviceGetAttribute(&gpu_numa_id, CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, cu_dev);
  if (gpu_numa_id < 0) gpu_numa_id = 0;

  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, gpu_id));

  char hostname[256] = {};
  gethostname(hostname, sizeof(hostname));

  if (rank == 0) {
    printf("================================================================\n");
    printf("  NCCL Send/Recv: cross-node GPU→CPU  (HBM vs Pinned vs EGM)\n");
    printf("  4 nodes × 1 GPU(GPU0) per node = 4 ranks\n");
    printf("================================================================\n\n");
  }

  printf("  R%d: node=%s, GPU%d (%s), GPU_NUMA=%d\n",
         rank, hostname, gpu_id, prop.name, gpu_numa_id);
  fflush(stdout);
  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

  // NCCL communicator
  ncclUniqueId nccl_id;
  if (rank == 0) NCCL_CHECK(ncclGetUniqueId(&nccl_id));
  MPI_CHECK(MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD));

  ncclComm_t nccl_comm;
  NCCL_CHECK(ncclCommInitRank(&nccl_comm, world_size, nccl_id, rank));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  if (rank == 0) printf("\n  NCCL communicator initialized.\n");

  const int numa_id = 0;

  // Case 1: Node{0,1} GPU0 send(HBM) → Node{2,3} recv(varies: HBM/Pinned/EGM)
  {
    int senders[2]  = {0, 1};
    int recvers[2]  = {2, 3};
    run_case("Case 1: Node{0,1} GPU0 HBM  -->  Node{2,3} recv buf varies",
             senders, recvers, rank, gpu_id, numa_id, /*vary_send=*/false, nccl_comm, stream);
  }

  // Case 2: Node{2,3} send(varies: HBM/Pinned/EGM) → Node{0,1} GPU0 recv(HBM)
  {
    int senders[2]  = {2, 3};
    int recvers[2]  = {0, 1};
    run_case("Case 2: Node{2,3} send buf varies  -->  Node{0,1} GPU0 HBM",
             senders, recvers, rank, gpu_id, numa_id, /*vary_send=*/true, nccl_comm, stream);
  }

  if (rank == 0) {
    printf("\n  Legend:\n");
    printf("    A) HBM:    GPU HBM (cudaMalloc)\n");
    printf("    B) Pinned: NUMA-pinned host (mmap+mbind(NUMA %d)+cudaHostRegister)\n", numa_id);
    printf("    C) EGM:    EGM on NUMA %d (cuMemCreate HOST_NUMA, GPU-mapped)\n", numa_id);
    printf("    Case 1: send=HBM(fixed), recv=A/B/C(varies)\n");
    printf("    Case 2: send=A/B/C(varies), recv=HBM(fixed)\n");
    printf("    GB/s = unidirectional BW = data_bytes / avg_time\n");
    printf("    avg_us = average latency per operation (microseconds)\n\n");
    fflush(stdout);
  }

  ncclCommDestroy(nccl_comm);
  cudaStreamDestroy(stream);
  MPI_CHECK(MPI_Finalize());
  return 0;
}
