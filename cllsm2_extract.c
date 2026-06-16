#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "libllsm2/llsm.h"

static int read_f32(const char* path, float** out, int* nout) {
  FILE* fp = fopen(path, "rb");
  if(!fp) return -1;
  if(fseek(fp, 0, SEEK_END) != 0) { fclose(fp); return -1; }
  long bytes = ftell(fp);
  if(bytes < 0) { fclose(fp); return -1; }
  rewind(fp);
  int n = (int)(bytes / (long)sizeof(float));
  float* x = (float*)malloc((size_t)n * sizeof(float));
  if(!x) { fclose(fp); return -1; }
  if(fread(x, sizeof(float), (size_t)n, fp) != (size_t)n) {
    free(x);
    fclose(fp);
    return -1;
  }
  fclose(fp);
  *out = x;
  *nout = n;
  return 0;
}

static int read_f0_csv(const char* path, float** out, int* nout) {
  FILE* fp = fopen(path, "r");
  if(!fp) return -1;
  int cap = 4096;
  int n = 0;
  float* f0 = (float*)malloc((size_t)cap * sizeof(float));
  if(!f0) { fclose(fp); return -1; }
  char line[512];
  while(fgets(line, sizeof(line), fp)) {
    char* p = line;
    char* end = NULL;
    float a = strtof(p, &end);
    if(end == p) continue;
    float value = a;
    p = end;
    while(*p == ' ' || *p == '\t') p++;
    if(*p == ',') {
      p++;
      value = strtof(p, &end);
      if(end == p) continue;
    }
    if(n >= cap) {
      cap *= 2;
      float* next = (float*)realloc(f0, (size_t)cap * sizeof(float));
      if(!next) {
        free(f0);
        fclose(fp);
        return -1;
      }
      f0 = next;
    }
    f0[n++] = value;
  }
  fclose(fp);
  *out = f0;
  *nout = n;
  return 0;
}

static int write_csv72(const char* path, llsm_coder* coder, llsm_chunk* chunk, int nfrm, int order_spec, int order_bap) {
  FILE* fp = fopen(path, "w");
  if(!fp) return -1;
  int dim = order_spec + order_bap + 3;
  for(int i = 0; i < nfrm; i++) {
    FP_TYPE* enc = llsm_coder_encode(coder, chunk->frames[i]);
    if(!enc) {
      fclose(fp);
      return -1;
    }
    for(int j = 0; j < dim; j++) {
      if(j) fputc(',', fp);
      fprintf(fp, "%.9g", (double)enc[j]);
    }
    fputc('\n', fp);
    free(enc);
  }
  fclose(fp);
  return 0;
}

int main(int argc, char** argv) {
  if(argc < 21) {
    fprintf(stderr,
      "usage: %s audio.f32 f0.csv out72.csv sample_rate hop_samples nfft maxnhar maxnhar_e npsd nchannel chanfreq0 chanfreq1 chanfreq2 order_spec order_bap rel_winsize lip_radius f0_refine hm_method frames\n",
      argv[0]);
    return 2;
  }
  const char* audio_path = argv[1];
  const char* f0_path = argv[2];
  const char* out_path = argv[3];
  int sample_rate = atoi(argv[4]);
  int hop_samples = atoi(argv[5]);
  int nfft = atoi(argv[6]);
  int maxnhar = atoi(argv[7]);
  int maxnhar_e = atoi(argv[8]);
  int npsd = atoi(argv[9]);
  int nchannel = atoi(argv[10]);
  float chanfreq0 = (float)atof(argv[11]);
  float chanfreq1 = (float)atof(argv[12]);
  float chanfreq2 = (float)atof(argv[13]);
  int order_spec = atoi(argv[14]);
  int order_bap = atoi(argv[15]);
  float rel_winsize = (float)atof(argv[16]);
  float lip_radius = (float)atof(argv[17]);
  int f0_refine = atoi(argv[18]);
  int hm_method = atoi(argv[19]);
  int frames = atoi(argv[20]);

  float* audio = NULL;
  float* f0 = NULL;
  int nx = 0;
  int nfrm = 0;
  if(read_f32(audio_path, &audio, &nx) != 0) {
    fprintf(stderr, "failed to read audio f32: %s\n", audio_path);
    return 1;
  }
  if(read_f0_csv(f0_path, &f0, &nfrm) != 0) {
    fprintf(stderr, "failed to read f0 csv: %s\n", f0_path);
    free(audio);
    return 1;
  }
  if(frames > 0 && frames < nfrm) nfrm = frames;
  if(nfrm <= 0) {
    fprintf(stderr, "empty f0\n");
    free(audio);
    free(f0);
    return 1;
  }

  llsm_aoptions* opt = llsm_create_aoptions();
  opt->thop = (FP_TYPE)hop_samples / (FP_TYPE)sample_rate;
  opt->maxnhar = maxnhar;
  opt->maxnhar_e = maxnhar_e;
  opt->npsd = npsd;
  opt->nchannel = nchannel;
  opt->chanfreq[0] = chanfreq0;
  opt->chanfreq[1] = chanfreq1;
  opt->chanfreq[2] = chanfreq2;
  opt->lip_radius = lip_radius;
  opt->f0_refine = f0_refine;
  opt->hm_method = hm_method;
  opt->rel_winsize = rel_winsize;

  llsm_chunk* chunk = llsm_analyze(opt, audio, nx, (FP_TYPE)sample_rate, f0, nfrm, NULL);
  if(!chunk) {
    fprintf(stderr, "llsm_analyze failed\n");
    llsm_delete_aoptions(opt);
    free(audio);
    free(f0);
    return 1;
  }
  llsm_chunk_tolayer1(chunk, nfft);
  llsm_coder* coder = llsm_create_coder(chunk->conf, order_spec, order_bap);
  if(write_csv72(out_path, coder, chunk, nfrm, order_spec, order_bap) != 0) {
    fprintf(stderr, "failed to write csv: %s\n", out_path);
    llsm_delete_coder(coder);
    llsm_delete_chunk(chunk);
    llsm_delete_aoptions(opt);
    free(audio);
    free(f0);
    return 1;
  }
  llsm_delete_coder(coder);
  llsm_delete_chunk(chunk);
  llsm_delete_aoptions(opt);
  free(audio);
  free(f0);
  return 0;
}
