#ifndef _GGFFT_H
#define _GGFFT_H

typedef struct {
  /* set by user by passing to init_plan */
  int dim;
  int *flengths;
  int *nprocl;
  int *ixor;
  int deg;

  /* set automatically in init_plan */
  int fftdim;
  int *fftdir;
  int *nfftdir;
  int *proc_coords;
  int *nbasis;
  int proc_id;
  int volproc;
  int totproc;
  MPI_Comm cart_comm;

  /* service (can change in the fft call) */
  int *lengths;
  int *cbasis;
  int **label;
  int *label_;
  int *kk;
  int *pp;

  _Complex double ***Mail;
  _Complex double **mail_;
  _Complex double *mail__;
} gg_plan;

void gg_init_plan(gg_plan *plan, int dim, int *flengths, int *nprocl, int *ixor, int deg, MPI_Comm old_comm);
void gg_distributed_multidim_fft(gg_plan *plan, int fftflag, _Complex double * vv);
void gg_destroy_plan(gg_plan *plan);

#endif
