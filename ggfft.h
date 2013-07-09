#ifndef _GGFFT_H
#define _GGFFT_H


void gg_init(int dim, int *flengths, int *nprocl, int *ixor);
void gg_distributed_multidim_fft(int fftflag, _Complex double * vv, int deg);
void gg_finalize();

#endif
