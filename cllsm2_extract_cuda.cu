/*
  cllsm2_extract_cuda — CUDA-accelerated LLSM2 parameter extraction

  Single translation unit: includes library .c files after providing
  cuFFT replacements for cdft/rdft/ddct (Ooura FFT).

  Build (Windows):
    nvcc -O3 -DFP_TYPE=float -I include -o build\cllsm2_extract_cuda.exe cllsm2_extract_cuda.cu -lcufft -lcudart
*/

#define _USE_MATH_DEFINES
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cufft.h>

#define CUDA_CHECK(x)  do { cuda_check((x), __FILE__, __LINE__); } while(0)
#define CUFFT_CHECK(x) do { cufft_check((x), __FILE__, __LINE__); } while(0)

static void cuda_check(cudaError_t e, const char* f, int l) {
  if(e != cudaSuccess) { fprintf(stderr, "CUDA %d at %s:%d: %s\n", e, f, l, cudaGetErrorString(e)); exit(1); }
}
static void cufft_check(cufftResult e, const char* f, int l) {
  if(e != CUFFT_SUCCESS) { fprintf(stderr, "cuFFT %d at %s:%d\n", e, f, l); exit(1); }
}

/* ---------- cuFFT plan cache ---------- */
#define MAX_PLANS 64
typedef struct { int n; int isgn; int type; cufftHandle p; int ok; } plan_entry;
static plan_entry plans[MAX_PLANS];
static int nplans;

static cufftHandle get_plan(int n, int isgn, int type) {
  for(int i = 0; i < nplans; i++)
    if(plans[i].n == n && plans[i].isgn == isgn && plans[i].type == type)
      return plans[i].p;
  if(nplans >= MAX_PLANS) { fprintf(stderr, "plan cache overflow\n"); exit(1); }
  cufftHandle p;
  if(type == 0)      CUFFT_CHECK(cufftPlan1d(&p, n, CUFFT_C2C, 1));
  else if(type == 1) CUFFT_CHECK(cufftPlan1d(&p, n, CUFFT_R2C, 1));
  else               CUFFT_CHECK(cufftPlan1d(&p, n, CUFFT_C2R, 1));
  plans[nplans].n = n; plans[nplans].isgn = isgn; plans[nplans].type = type;
  plans[nplans].p = p; plans[nplans].ok = 1; nplans++;
  return p;
}

/* ---------- GPU buffer pool ---------- */
static float* gpubuf;
static int    gpucap;

static float* gpialloc(int nf) {
  if(gpucap < nf) {
    if(gpubuf) cudaFree(gpubuf);
    CUDA_CHECK(cudaMalloc(&gpubuf, (size_t)nf * sizeof(float)));
    gpucap = nf;
  }
  return gpubuf;
}

static void cleanup_gpu(void) {
  if(gpubuf) { cudaFree(gpubuf); gpubuf = NULL; gpucap = 0; }
  for(int i = 0; i < nplans; i++)
    if(plans[i].ok) { cufftDestroy(plans[i].p); plans[i].ok = 0; }
  nplans = 0;
}

/* ================================================================== *
 *  Ooura FFT is provided by fftsg_h.c (included below).              *
 *  GPU is used ONLY for the batched harmonic analysis in ha_cuda.     *
 *  Individual FFT calls (~20K+ per file) stay on CPU via Ooura,       *
 *  avoiding ~40K+ cudaMemcpy round-trips per inference.               *
 * ================================================================== */

/* ================================================================== *
 *  CUDA kernel: CZT for one frame                                     *
 * ================================================================== */
__global__ void czt_frame_kernel(const float* x, float* re, float* im,
                                  int nx, int nhar, float omega0) {
  int h = threadIdx.x;
  if(h > nhar) return;
  float sr = 0, si = 0;
  int shift = nx / 2;
  for(int i = 0; i < nx; i++) {
    float d = (float)(i - shift);
    float phi = omega0 * ((float)h * i + d * d * 0.5f);
    float wr = __cosf(phi), wi = __sinf(phi);
    sr += x[i] * wr;
    si += x[i] * wi;
  }
  re[h] = sr;
  im[h] = si;
}

/* ----- GPU kernel: extract frame + apply Blackman window ----- */
__global__ void extract_window_kernel(const float* x, int nx,
    const int* centers, const int* winsizes,
    float* dst, int nfft, int nfrm) {
  int i = blockIdx.y;
  int j = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= nfrm || j >= nfft) return;
  int ws = winsizes[i];
  int ct = centers[i];
  float* out = dst + (size_t)i * nfft;
  if (j < ws) {
    int is = ct + j - ws / 2;
    float a = 2.0f * (float)M_PI * j / ws;
    float w = 0.42f - 0.5f * cosf(a) + 0.08f * cosf(2.0f * a);
    out[j] = (is >= 0 && is < nx) ? x[is] * w : 0.0f;
  } else {
    out[j] = 0.0f;
  }
}

/* GPU CZT for one frame: windowed frame x[nx], return nhar ampl/phse */
static void gpu_czt(float* x, int nx, float f0, float fs, int nhar,
                     float* ampl, float* phse) {
  float omega0 = 2.0f * (float)M_PI * f0 / fs;
  int need = nx + 2 * (nhar + 1);
  float* pool = gpialloc(need);
  float* dx = pool;
  float* dre = pool + nx;
  float* dim = pool + nx + (nhar + 1);
  CUDA_CHECK(cudaMemcpy(dx, x, (size_t)nx * sizeof(float), cudaMemcpyHostToDevice));
  int nth = nhar + 1 < 256 ? nhar + 1 : 256;
  czt_frame_kernel<<<1, nth>>>(dx, dre, dim, nx, nhar, omega0);
  float hr[2048], hi[2048];
  int m = nhar + 1 > 2048 ? 2048 : nhar + 1;
  CUDA_CHECK(cudaMemcpy(hr, dre, (size_t)m * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(hi, dim, (size_t)m * sizeof(float), cudaMemcpyDeviceToHost));

  /* window normalization */
  float ws = 0;
  for(int i = 0; i < nx; i++) {
    float a = 2.0f * (float)M_PI * i / nx;
    float w = 0.42f - 0.5f * cosf(a) + 0.08f * cosf(2.0f * a);
    ws += w;
  }
  for(int h = 0; h < nhar; h++) {
    float ishift = (float)(nx / 2) * omega0 * (h + 1.0f);
    float cr = cosf(ishift), si = sinf(ishift);
    float dr = hr[h+1] * cr - hi[h+1] * si;
    float di = hr[h+1] * si + hi[h+1] * cr;
    ampl[h] = sqrtf(dr*dr + di*di) * 2.0f / ws;
    phse[h] = atan2f(di, dr);
  }
}

/* ================================================================== *
 *  Include library source files                                       *
 * ================================================================== */

#include "ciglet/ciglet.c"
#include "ciglet/external/fftsg_h.c"
#include "ciglet/external/fast_median.c"

#include "libllsm2/llsm.h"
#include "libllsm2/constants.h"
#include "libllsm2/buffer.h"
#include "libllsm2/container.c"
#include "libllsm2/frame.c"
#include "libllsm2/dsputils.c"
#include "libllsm2/llsmutils.c"

/* ================================================================== *
 *  GPU harmonic analysis — replaces llsm_harmonic_analysis           *
 *  Defined before layer0.c so llsm_analyze calls this instead.       *
 * ================================================================== */

static void ha_cuda(FP_TYPE* x, int nx, FP_TYPE fs, FP_TYPE* f0, int nfrm,
                     FP_TYPE thop, FP_TYPE rw, int mxh, int method,
                     int* dn, FP_TYPE** da, FP_TYPE** dp) {
  if(method == 1) {
    for(int i = 0; i < nfrm; i++) {
      if(f0[i] <= 0) continue;
      int nhar = (int)(fs / f0[i] / 2); if(nhar > mxh) nhar = mxh;
      dn[i] = nhar;
      int ws = (int)(fs / f0[i] * rw / 2) * 2;
      int ct = (int)(i * thop * fs);
      float* frm = fetch_frame(x, nx, ct, ws);
      float* w = blackman(ws);
      for(int j = 0; j < ws; j++) frm[j] *= w[j];
      free(w);
      da[i] = (FP_TYPE*)calloc((size_t)nhar, sizeof(FP_TYPE));
      dp[i] = (FP_TYPE*)calloc((size_t)nhar, sizeof(FP_TYPE));
      gpu_czt(frm, ws, f0[i], fs, nhar, da[i], dp[i]);
      free(frm);
    }
  } else {
    int nfft = llsm_get_fftsize(f0, nfrm, fs, rw);
    int ns = nfft / 2 + 1;
    int nv = 0; for(int i = 0; i < nfrm; i++) if(f0[i] > 0) nv++;
    if(!nv) return;
    int* iv = (int*)calloc((size_t)nv, sizeof(int));
    int* wz = (int*)calloc((size_t)nv, sizeof(int));
    int* ct = (int*)calloc((size_t)nv, sizeof(int));
    for(int i = 0, v = 0; i < nfrm; i++)
      if(f0[i] > 0) { iv[v] = i; wz[v] = (int)(fs / f0[i] * rw / 2) * 2; ct[v] = (int)(i * thop * fs); v++; }

    /* Precompute window sums on CPU (exact Blackman sum = 0.42 * winsize) */
    float* wsum = (float*)calloc((size_t)nv, sizeof(float));
    for(int v = 0; v < nv; v++) wsum[v] = 0.42f * wz[v];

    /* Upload audio + frame metadata to GPU */
    float *dx, *dfrm, *dout;
    int *dct, *dwz;
    size_t nfft_sz = (size_t)nfft;
    CUDA_CHECK(cudaMalloc(&dx,   (size_t)nx * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dfrm, (size_t)nv * nfft_sz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dout, (size_t)nv * (size_t)ns * 2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dct,  (size_t)nv * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dwz,  (size_t)nv * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(dx,  x,  (size_t)nx * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dct, ct, (size_t)nv * sizeof(int),   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dwz, wz, (size_t)nv * sizeof(int),   cudaMemcpyHostToDevice));

    /* GPU: extract + window all frames in one kernel launch */
    {
      int bpf = (nfft + 255) / 256;
      dim3 grid_dim(bpf, nv);
      extract_window_kernel<<<grid_dim, 256>>>(dx, nx, dct, dwz, dfrm, nfft, nv);
    }

    /* Batched cuFFT R2C (all frames, same nfft) */
    cufftHandle pl;
    CUFFT_CHECK(cufftPlan1d(&pl, nfft, CUFFT_R2C, nv));
    CUFFT_CHECK(cufftExecR2C(pl, dfrm, (cufftComplex*)dout));

    /* Download all spectrograms in one transfer */
    FP_TYPE** mg = (FP_TYPE**)malloc2d_((size_t)nv, (size_t)ns, sizeof(FP_TYPE));
    FP_TYPE** ph = (FP_TYPE**)malloc2d_((size_t)nv, (size_t)ns, sizeof(FP_TYPE));

    float* ho = (float*)malloc((size_t)nv * (size_t)ns * 2 * sizeof(float));
    CUDA_CHECK(cudaMemcpy(ho, dout, (size_t)nv * (size_t)ns * 2 * sizeof(float), cudaMemcpyDeviceToHost));

    for(int t = 0; t < nv; t++) {
      float norm = 512.0f / wsum[t];
      float* fo = ho + (size_t)t * (size_t)ns * 2;
      for(int k = 0; k < ns; k++) {
        float re = fo[k*2], im = fo[k*2+1];
        mg[t][k] = sqrtf(re*re + im*im) * norm;
        ph[t][k] = atan2f(im, re);
      }
    }

    CUDA_CHECK(cudaFree(dx)); CUDA_CHECK(cudaFree(dfrm));
    CUDA_CHECK(cudaFree(dout)); CUDA_CHECK(cudaFree(dct)); CUDA_CHECK(cudaFree(dwz));
    CUFFT_CHECK(cufftDestroy(pl));
    free(ho);

    for(int v = 0; v < nv; v++) {
      int idx = iv[v];
      for(int j = 0; j < ns; j++) mg[v][j] = logf(mg[v][j] + 1e-8f);
      int nhar = (int)(fs / f0[idx] / 2); if(nhar > mxh) nhar = mxh;
      dn[idx] = nhar;
      da[idx] = (FP_TYPE*)calloc((size_t)nhar, sizeof(FP_TYPE));
      dp[idx] = (FP_TYPE*)calloc((size_t)nhar, sizeof(FP_TYPE));
      for(int h = 0; h < nhar; h++) {
        float fh = f0[idx] * (h + 1.0f);
        int li = (int)(fh * 0.7f / fs * nfft); if(li < 1) li = 1;
        int ui = (int)(fh * 1.3f / fs * nfft); if(ui > ns-1) ui = ns-1;
        int pk = li; float pv = mg[v][li];
        for(int k = li+1; k <= ui; k++) if(mg[v][k] > pv) { pv = mg[v][k]; pk = k; }
        float a = mg[v][pk-1], b = mg[v][pk], c = mg[v][pk+1];
        float a1 = (a + c) * 0.5f - b, a2 = c - b - a1, xo = -a2 / a1 * 0.5f;
        if(fabsf(xo) > 1) xo = 0;
        float pa = a1 * xo * xo + a2 * xo + b;
        if(pa > b + 0.2f) pa = b + 0.2f;
        da[idx][h] = expf(pa);
        float fi = pk + xo; int fii = (int)fi; float fr = fi - fii;
        dp[idx][h] = ph[v][fii] * (1 - fr) + ph[v][fii + 1] * fr;
      }
    }
    free2d_((void**)mg, (size_t)nv); free2d_((void**)ph, (size_t)nv);
    free(iv); free(wz); free(ct); free(wsum);
  }
}

#define llsm_harmonic_analysis ha_cuda

/* ================================================================== *
 *  Include remaining library sources                                  *
 * ================================================================== */

#include "libllsm2/layer0.c"
#include "libllsm2/layer1.c"
#include "libllsm2/coder.c"

/* ================================================================== *
 *  I/O helpers                                                        *
 * ================================================================== */

static int read_f32(const char* path, float** out, int* n) {
  FILE* f = fopen(path, "rb"); if(!f) return -1;
  fseek(f, 0, SEEK_END); long bytes = ftell(f); rewind(f);
  int nn = (int)(bytes / 4);
  float* x = (float*)malloc((size_t)nn * sizeof(float));
  if(!x || fread(x, sizeof(float), (size_t)nn, f) != (size_t)nn) { free(x); fclose(f); return -1; }
  fclose(f); *out = x; *n = nn; return 0;
}

static int read_f0_csv(const char* path, float** out, int* n) {
  FILE* f = fopen(path, "r"); if(!f) return -1;
  int cap = 4096, nn = 0;
  float* f0 = (float*)malloc((size_t)cap * sizeof(float));
  if(!f0) { fclose(f); return -1; }
  char line[512];
  while(fgets(line, sizeof(line), f)) {
    char *p = line, *ep;
    float v = strtof(p, &ep); if(ep == p) continue;
    p = ep; while(*p == ' ' || *p == '\t') p++;
    if(*p == ',') { p++; v = strtof(p, &ep); }
    if(nn >= cap) { cap *= 2; float* r = (float*)realloc(f0, (size_t)cap * sizeof(float)); if(!r) { free(f0); fclose(f); return -1; } f0 = r; }
    f0[nn++] = v;
  }
  fclose(f); *out = f0; *n = nn; return 0;
}

static int write_csv72(const char* path, llsm_coder* c, llsm_chunk* h, int nf,
                        int order_spec, int order_bap) {
  FILE* f = fopen(path, "w"); if(!f) return -1;
  int nd = order_spec + order_bap + 3;
  for(int i = 0; i < nf; i++) {
    FP_TYPE* e = llsm_coder_encode(c, h->frames[i]);
    if(!e) { fclose(f); return -1; }
    for(int j = 0; j < nd; j++) {
      if(j) fputc(',', f);
      fprintf(f, "%.9g", (double)e[j]);
    }
    fputc('\n', f);
    free(e);
  }
  fclose(f); return 0;
}

/* ================================================================== *
 *  main                                                               *
 * ================================================================== */
int main(int argc, char** argv) {
  int ndev; CUDA_CHECK(cudaGetDeviceCount(&ndev));
  if(!ndev) { fprintf(stderr, "No CUDA device found.\n"); return 1; }
  cudaSetDevice(0);

  fprintf(stderr, "cllsm2_extract_cuda (CUDA accelerated)\n");

  if(argc < 21) {
    fprintf(stderr, "usage: %s audio.f32 f0.csv out72.csv sr hop nfft mxh mxhe npsd nch cf0 cf1 cf2 os ob rw lr frf hm frm\n", argv[0]);
    return 2;
  }

  float *audio, *f0csv;
  int nx, nf;
  if(read_f32(argv[1], &audio, &nx)) { fprintf(stderr, "read audio failed\n"); return 1; }
  if(read_f0_csv(argv[2], &f0csv, &nf)) { fprintf(stderr, "read f0 failed\n"); free(audio); return 1; }

  int limit = atoi(argv[20]);
  if(limit > 0 && limit < nf) nf = limit;
  if(nf <= 0) { free(audio); free(f0csv); return 1; }

  llsm_aoptions* opts = llsm_create_aoptions();
  opts->thop       = (FP_TYPE)atof(argv[5]) / (FP_TYPE)atof(argv[4]);
  opts->maxnhar    = atoi(argv[7]);
  opts->maxnhar_e  = atoi(argv[8]);
  opts->npsd       = atoi(argv[9]);
  opts->nchannel   = atoi(argv[10]);
  opts->chanfreq[0]= (FP_TYPE)atof(argv[11]);
  opts->chanfreq[1]= (FP_TYPE)atof(argv[12]);
  opts->chanfreq[2]= (FP_TYPE)atof(argv[13]);
  opts->lip_radius = (FP_TYPE)atof(argv[17]);
  opts->f0_refine  = atoi(argv[18]);
  opts->hm_method  = atoi(argv[19]);
  opts->rel_winsize= (FP_TYPE)atof(argv[16]);

  llsm_coder* coder = NULL;
  llsm_chunk* chunk = llsm_analyze(opts, audio, nx, (FP_TYPE)atof(argv[4]),
                                    f0csv, nf, NULL);
  if(!chunk) { fprintf(stderr, "analysis failed\n"); cleanup_gpu(); goto cleanup1; }

  llsm_chunk_tolayer1(chunk, atoi(argv[6]));
  coder = llsm_create_coder(chunk->conf, atoi(argv[14]), atoi(argv[15]));
  if(write_csv72(argv[3], coder, chunk, nf, atoi(argv[14]), atoi(argv[15]))) {
    fprintf(stderr, "write output failed\n");
    llsm_delete_coder(coder); llsm_delete_chunk(chunk); cleanup_gpu(); goto cleanup1;
  }

  llsm_delete_coder(coder);
  llsm_delete_chunk(chunk);
  cleanup_gpu();

cleanup1:
  llsm_delete_aoptions(opts);
  free(audio);
  free(f0csv);
  return 0;
}
