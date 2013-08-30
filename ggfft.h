#ifndef _GGFFT_H
#define _GGFFT_H

typedef struct {
  /* set by user by passing to init_plan */
  int dim;
  int *flengths;
  int *nprocl;
  int *ixor;

  /* set automatically in init_plan */
  int fftdim;
  int *fftdir;
  int *nfftdir;
  int *proc_coords;
  int *lengths;
  int *cbasis;
  int *nbasis;
  int proc_id;
  int volproc;
  int totproc;

  /* service */
  int **label;
  int *label_;
  int *kk;
  int *pp;

  _Complex double ***Mail;
  _Complex double **mail_;
  _Complex double *mail__;
} gg_plan;

void gg_init_plan(gg_plan *plan, int dim, int *flengths, int *nprocl, int *ixor, int deg);
void gg_distributed_multidim_fft(gg_plan *plan, int fftflag, _Complex double * vv, int deg);
void gg_destroy_plan(gg_plan *plan);

#endif
