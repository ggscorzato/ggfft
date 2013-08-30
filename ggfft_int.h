#ifndef _GGFFT_INT_H
#define _GGFFT_INT_H


/***** DECLARATION OF FUNCTIONS USED ONLY INTERNALLY (THOSE AVAILABLE TO USER ARE IN ggfft.h) *****/
void transpose(gg_plan *plan, _Complex double * vv, int * permutin, int *permutout);
void serial_multidim_fft(_Complex double * cdata, int fdeg, int * nn, int dim_sfft, int isign);

#endif
