#ifndef _GGFFT_INT_H
#define _GGFFT_INT_H

/*****
      GLOBAL VARIABLES SET BY USER:
      _dim: dimension of the data 
      _flengths[]: array of length dim;  it contains the full lengths of points foreach direction.
      _nprocl[]: array of length dim; it contains the number of processors foreach direction 
      _ixor[]: array of size dim containing the order (from fastest to slowest) in which the coordinates run in seq
      ind

      GLOBAL VARAIABLES SET BY INIT:
      _fftdim: dimension of the serial fft
      -fftdir[]: array of size _dim containing in the first _fftdim elements the non parallelized directions
      _proc_coords[]: array of length dim with the coordinates of this process 
      _proc_id: rank of this process 
      _volproc: number of points per process 
      _totproc: total number of processes

      GLOBAL VAIABLES FOR SERVICE:
      _label: service variables for sending
      _kk:    service variables for sending
*****/

int _dim, _volproc, _totproc, _fftdim, _proc_id;
int *_flengths, *_lengths, *_nprocl, *_ixor, *_cbasis, *_nbasis, *_fftdir, *_nfftdir,*_proc_coords;
int **_label, *_kk;
_Complex double ***_Mail,**__mail, *___mail;

/***** DECLARATION OF FUNCTIONS USED ONLY INTERNALLY (THOSE AVAILABLE TO USER ARE IN ggfft.h) *****/
void permute(_Complex double * vv, int deg, int * permutin, int * permutout);
void serial_multidim_fft(_Complex double * cdata, int fdeg, int * nn, int dim_sfft, int isign);



#endif
