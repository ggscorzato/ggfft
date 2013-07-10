#include <stdlib.h>
#include <stdio.h>
#include <complex.h>
#include <math.h>
#include <mpi.h>
#include "ggfft_int.h"
#include "ggfft.h"

#define DEBUG 1
/*****
      Initialization routine. It should be called after MPI_Init, but before any function in ggfft.
      dim: dimension of the data to be FFT,
      flengths[]: array of length dim;  it contains the full lengths of points foreach direction.
      nprocl[]: array of length dim; it contains the number of processors foreach direction.
      ixor[]: array of length dim; ixor[0] is the fastest direction, ixor[dim-1] the slowest in lexicografic order.
      the three pointers above should be already allocated before passing them to gg_init.
*****/
void gg_init(int dim, int *flengths, int *nprocl, int *ixor){

  int mu,i,*periods,*isfft,volproc2,check;
  MPI_Comm cart_comm;

  /*** copy the user set variables to global variables ***/
  _dim=dim;
  _flengths=flengths;
  _nprocl=nprocl;
  _ixor=ixor;

  /*** define coordinates of this process w.r.t. the whole lattice, both sequential (_proc_id) and vector
     (_proc_coords) ***/
  _proc_coords=calloc(_dim,sizeof(int));
  periods=calloc(_dim,sizeof(int));
  for(mu=0;mu<_dim;mu++) periods[mu]=1; // torus
  MPI_Cart_create(MPI_COMM_WORLD, _dim, _nprocl, periods, 1, &cart_comm);
  MPI_Comm_rank(cart_comm,&_proc_id);
  MPI_Cart_coords(cart_comm,_proc_id,_dim,_proc_coords);
  MPI_Comm_size(cart_comm,&check);
  _totproc=1;
  for(mu=0;mu<_dim;mu++) _totproc*=_nprocl[mu];
  if(check!=_totproc){
    if(_proc_id==0){ 
      fprintf(stderr,"ERROR: the number or processes defined by the lattice (%d) does not match with the one" 
	      "set by mpirun(%d).\n",_totproc,check);
    }
    exit(-1);
  }

  /* basis for the conversion of indices between sequential and vector. _nbasis should be the one used by
     MPI_Cart_coords. This assumes the one implemented in openmpi */
  _nbasis=calloc(_dim,sizeof(int)); 
  _nbasis[_dim-1]=1;  
  for(mu=_dim-2; mu>=0;mu--) _nbasis[mu]=_nbasis[mu+1]*_nprocl[mu+1];
  check=0;
  for(mu=0;mu<_dim;mu++) check+=_nbasis[mu]*_proc_coords[mu];
  if(check!=_proc_id){
    fprintf(stderr,"ERROR: processors mapping error in proc: %d. It does not match with: %d\n",_proc_id,check);
    exit(-1);
  }
  
  /*** define local coordinates ***/
  _lengths=calloc(_dim,sizeof(int));
  _volproc=1;
  for(mu=0;mu<_dim;mu++){
    _lengths[mu]=_flengths[mu]/_nprocl[mu]; // local lengths of the portion of lattice in a process.
    _volproc*=_lengths[mu]; // local volume
  }
  _cbasis=calloc(_dim,sizeof(int));
  _cbasis[_ixor[0]]=1;
  for(mu=1;mu<_dim;mu++) _cbasis[_ixor[mu]]=_cbasis[_ixor[mu-1]]*_lengths[_ixor[mu-1]];
  volproc2=_volproc/_totproc;  /* fraction of volume going from any proc to any other */

  /*** determine the directions and the dimension of the serial FFT's ***/
  _fftdir=calloc(_dim,sizeof(int));
  _nfftdir=calloc(_dim,sizeof(int));
  isfft=calloc(_dim,sizeof(int));
  for(mu=0;mu<_dim;mu++) _fftdir[mu]=-1;
  for(mu=0;mu<_dim;mu++) isfft[mu]=0;
  _fftdim=0;
  mu=0;
  while(_nprocl[_ixor[mu]]>1) mu++;
  _fftdir[0]=_ixor[mu];
  isfft[_ixor[mu]]=1;
  _fftdim=1;
  if(_nprocl[_ixor[mu+1]]==1 &&  _dim%2==0 && _dim>mu+1){
    _fftdir[1]=_ixor[mu+1];
    isfft[_ixor[mu+1]]=1;
    _fftdim=2;
    if(_nprocl[_ixor[mu+2]]==1 && _nprocl[_ixor[mu+3]]==1 && _dim%4==0 && _dim>mu+3){
      _fftdir[2]=_ixor[mu+2];
      _fftdir[3]=_ixor[mu+3];
      isfft[_ixor[mu+2]]=1;
      isfft[_ixor[mu+3]]=1;
      _fftdim=4;
    } else if(_nprocl[_ixor[mu+2]]==1 && _dim%3==0 && _dim>mu+2){
      _fftdir[2]=_ixor[mu+2];
      isfft[_ixor[mu+2]]=1;
      _fftdim=3;
    }
  }
  if(_fftdim==0){
    if(_proc_id==0) fprintf(stderr,"ERROR: at least one direction should not be distributed.\n");
    exit(-1);
  }
  i=0;
  for(mu=0;mu<_dim;mu++){
    if(!isfft[mu]){
      _nfftdir[i] = mu;
    }
  }
  /*** allocate labels and indices for exchange vectors ***/
  _kk = calloc(_totproc,sizeof(int));
  _label = calloc(_totproc,sizeof(int*));
  for(i=0;i<_totproc;i++){
    _label[i] = calloc(volproc2,sizeof(int));
  }
  free(periods);
  free(isfft);
}

/***** finalize function to be called at the end to free memory *****/
void gg_finalize(){
  free(_flengths);
  free(_lengths);
  free(_nprocl);
  free(_ixor);
  free(_proc_coords);
  free(_nbasis);
  free(_cbasis);
  free(_fftdir);
  free(_nfftdir);
  free(_kk);
  free(_label);
}

/*****  distributed multidimensional FFT.
	fftflag: 1 direct FFT, -1 inverse FFT
	vv: array of size= deg * (\prod_mu _flengths[mu]). This vector will be fft'd.
	deg: internal degeneracy of the data in vv. 
*****/

void gg_distributed_multidim_fft(int fftflag, _Complex double * vv, int deg){

  int i,j,nfftpoints,repl_slow,repl_fast,mu,it,jj,nu,volproc2;
  int *nn, *inperm, *outperm;
  _Complex double *** Mail,* pnt0;

  /* allocate exchange vectors */
  volproc2=_volproc/_totproc;
  Mail = malloc(_totproc*sizeof(_Complex double**));
  for(i=0;i<_totproc;i++){
    Mail[i] = malloc(volproc2*sizeof(_Complex double*));
    for(j=0;j<volproc2;j++){
      Mail[i][j] = malloc(deg*sizeof(_Complex double));
    }
  }

  inperm=calloc(_dim,sizeof(int));
  for(mu=0; mu<_dim;mu++) inperm[mu]=mu;
  outperm=calloc(_dim,sizeof(int));
  for(mu=0; mu<_dim;mu++) outperm[mu]=mu;
  nn=calloc(_fftdim,sizeof(int));
  for(mu=0; mu<_fftdim;mu++) nn[mu] = _flengths[_fftdir[mu]];
  for(mu=0;mu<_dim;mu++) _lengths[mu]=_flengths[mu]/_nprocl[mu];
  _cbasis[_ixor[0]]=1;
  for(mu=1;mu<_dim;mu++) _cbasis[_ixor[mu]]=_cbasis[_ixor[mu-1]]*_lengths[_ixor[mu-1]];
  repl_fast=_cbasis[_fftdir[0]];
  nfftpoints=1;
  for(mu=0;mu<_fftdim;mu++) nfftpoints*=_lengths[_fftdir[mu]];
  repl_slow=_volproc/(repl_fast*nfftpoints);

  for(it=0;it<_dim;it+=_fftdim){  /* loop on fft subspaces */

    for(jj=0;jj<repl_slow;jj++){  /* loop on all points in orthogonal directions */
      pnt0=vv+(jj*(nfftpoints)*repl_fast*deg);
      serial_multidim_fft(pnt0,repl_fast*deg,nn,_fftdim,fftflag);  /* fft on the LAST _fftdim directions */
    }
    
    if(it<_dim-_fftdim){                   /* transpose after fft, except the last time */
      for(mu=0; mu<_fftdim;mu++){
	outperm[_nfftdir[it+mu]] = inperm[_fftdir[mu]];
	outperm[_fftdir[mu]] = inperm[_nfftdir[it+mu]];
      }
      
      permute(vv,deg,inperm,outperm,Mail);
      
      for(mu=0; mu<_fftdim;mu++) nn[mu] = _flengths[outperm[_fftdir[mu]]];
      for(mu=0;mu<_dim;mu++) _lengths[mu]=_flengths[outperm[mu]]/_nprocl[mu];
      _cbasis[_ixor[0]]=1;
      for(mu=1;mu<_dim;mu++) _cbasis[_ixor[mu]]=_cbasis[_ixor[mu-1]]*_lengths[_ixor[mu-1]];
      repl_fast=_cbasis[_fftdir[0]];
      nfftpoints=1;
      for(mu=0;mu<_fftdim;mu++) nfftpoints*=_lengths[_fftdir[mu]];
      repl_slow=_volproc/(repl_fast*nfftpoints);
      
      for(nu=0; nu<_dim;nu++) inperm[nu]=outperm[nu];
    }
  }  /* endo of loop on fft subspaces */
  
  for(nu=0; nu<_dim;nu++) outperm[nu]=nu; /* set back to the original order. */
  permute(vv,deg,inperm,outperm,Mail);

  free(Mail);
  free(inperm);
  free(outperm);
  free(nn);
}

/*****
      Permutes the dim-dimensional array vv (linearly arranged), according to the permutation permutin ->
      permutout.  Two permutation argumens are useful because the routine makes reference also to variables
      (_nprocl, _proc_coords, _nbasis) that always refer to the original ordering and are never changed by a
      permutation. In fact, only the data in vv are permuted. No global variable is changed.
      vv[]: array of size= deg * (\prod_mu _flengths[mu]). This vector will be permuted. [I&O]
      permutin[]: permutation of the original coordinates assumed in input [I]
      permutout[]: permutation of the original coordinates found in output [O]
*****/

void permute(_Complex double * vv, int deg, int * permutin, int * permutout, _Complex double *** Mail){

  int seqin,seqout,mu,ind,i,j,ka,proc_out,volproc2;
  int *lleng_in,*lleng_out,*temp_orig,*locoIn,* glcoIn,*locoOut,* glcoOut,*proc_coordOut,*cbasisIn,*cbasisOut;
  MPI_Status mpistatus;

  lleng_in=calloc(_dim,sizeof(int));
  lleng_out=calloc(_dim,sizeof(int));
  cbasisIn=calloc(_dim,sizeof(int));
  cbasisOut=calloc(_dim,sizeof(int));
  locoIn=calloc(_dim,sizeof(int));
  locoOut=calloc(_dim,sizeof(int));
  glcoIn=calloc(_dim,sizeof(int));
  glcoOut=calloc(_dim,sizeof(int));
  proc_coordOut=calloc(_dim,sizeof(int));
  temp_orig=calloc(_dim,sizeof(int));

  volproc2=_volproc/_totproc;


  for(mu=0; mu<_dim; mu++){
    lleng_in[mu]=_flengths[permutin[mu]]/_nprocl[mu];
    lleng_out[mu]=_flengths[permutout[mu]]/_nprocl[mu];
  }

  cbasisIn[_ixor[0]]=1;
  cbasisOut[_ixor[0]]=1;
  for(mu=1;mu<_dim;mu++){
    cbasisIn[_ixor[mu]]=cbasisIn[_ixor[mu-1]]*lleng_in[_ixor[mu-1]];
    cbasisOut[_ixor[mu]]=cbasisOut[_ixor[mu-1]]*lleng_out[_ixor[mu-1]];
  }

  for(ind=0; ind < _totproc; ind++) _kk[ind]=0;

  for(i=0; i<_volproc; i++){    // main loop: for every local site

    seqin=i;
    for(mu=0; mu < _dim; mu++){   // construct local an global IN index (from the sequential)
      locoIn[mu]=(seqin / cbasisIn[mu]) % lleng_in[mu];
      glcoIn[mu]= locoIn[mu]+lleng_in[mu]*_proc_coords[mu];
    }

    // transpose the global indices glcoOut[mu] = glcoIn[po[pi^-1[mu]]]:
    for(mu=0; mu < _dim; mu++) temp_orig[permutin[mu]]=glcoIn[mu]; 
    for(mu=0; mu < _dim; mu++) glcoOut[mu]=temp_orig[permutout[mu]];
    proc_out=0;
    seqout=0;
    for(mu=0; mu < _dim; mu++){     // reconstruct local and sequential OUT index
      proc_coordOut[mu]=glcoOut[mu]/lleng_out[mu];
      locoOut[mu]=glcoOut[mu] % lleng_out[mu];
      proc_out+=proc_coordOut[mu]*_nbasis[mu];
      seqout+=locoOut[mu]*cbasisOut[mu];
    }
    for(j=0;j<deg;j++) Mail[proc_out][_kk[proc_out]][j]=vv[i*deg+j];  // store index and variable for expedition
    _label[proc_out][_kk[proc_out]]=seqout;
    _kk[proc_out]++;
  } // end of main loop on every local site.

  for(ind=0; ind < _totproc; ind++){  // Send
    if(ind!=_proc_id){
      MPI_Sendrecv_replace(Mail[ind][0],volproc2*deg,MPI_DOUBLE_COMPLEX,ind,0,ind,MPI_ANY_TAG,MPI_COMM_WORLD,
			   &mpistatus);
      MPI_Sendrecv_replace(_label[ind],volproc2,MPI_INT,ind,0,ind,MPI_ANY_TAG,MPI_COMM_WORLD,&mpistatus);
    }
  }
  for(ind=0; ind< _totproc; ind++){   // copy back the exchanged data onto the array vv
    for(ka=0; ka< volproc2; ka++){
      for(j=0;j<deg;j++){
	vv[_label[ind][ka]*deg+j]=Mail[ind][ka][j];
      }
    }
  }
  free(lleng_in);
  free(lleng_out);
  free(cbasisIn);
  free(cbasisOut);
  free(locoIn);
  free(locoOut);
  free(glcoIn);
  free(glcoOut);
  free(proc_coordOut);
  free(temp_orig);
}

/***** 
       Numerical Recipes function fourn, slightly modifyed to operate deg complex FFT at the same time.  Data are
       in cdata[0...(prod_nn*deg)-1]. The routine replaces cdata by its (dim_sfft)-dimensional discrete Fourier
       transform, if isign is input as 1.  nn[0...(dim_sfft)-1] is an integer array containing the lengths of each
       dimension (number of complex values), which MUST all be powers of 2. data is a real array of length twice
       the product of these lengths, in which the data are stored as in a multidimensional complex array: real and
       imaginary parts of each element are in consecutive locations, and the rightmost index of the array increases
       most rapidly as one proceeds along data.  For a two-dimensional array, this is equivalent to storing the
       array by rows. If isign is input as -1, data is replaced by its inverse transform times the product of the
       lengths of all dimensions.  call it with isign=-1 to have the same definition of matlab fft: F_k =
       \sum_j=0^N-1 exp[-i 2 \pi j k] f_k
*****/

void serial_multidim_fft(_Complex double * cdata, int fdeg, int * nn, int dim, int isign)
{
  int id,jd;
  unsigned long i1,i2,i3,i2rev,i3rev,ip1,ip2,ip3,ifp1,ifp2;
  unsigned long ibit,k1,k2,n,nprev,nrem,ntot;
  unsigned long j3,j3rev,h1,h2;
  _Complex double w,wp,ctemp,swap;
  double theta,wtemp;       //Double precision for trigonometric recurrences.
  for (ntot=1,id=0;id<dim;id++) ntot *= nn[id];   // Compute total number of sites.
  nprev=1;

  for (id=dim-1;id>=0;id--) {           // Main loop over the dimensions.
    n=nn[id];
    nrem=ntot/(n*nprev);
    ip1=nprev << 1;
    ip2=ip1*n;
    ip3=ip2*nrem;
    i2rev=1;
    for (i2=1;i2<=ip2;i2+=ip1) {      //This is the bit-reversal section of the routine. 
      if (i2 < i2rev) {
	for (i1=i2;i1<=i2+ip1-2;i1+=2) {
	  for (i3=i1;i3<=ip3;i3+=ip2) {
	    i3rev=i2rev+i3-i2;
	    j3=i3>>1;
	    j3rev=i3rev>>1;
	    for(jd=0;jd<fdeg;jd++){
	      swap=cdata[j3*fdeg+jd];
	      cdata[j3*fdeg+jd]=cdata[j3rev*fdeg+jd];
	      cdata[j3rev*fdeg+jd]=swap;
	    }
	  }
	}
      }
      ibit=ip2 >> 1;
      while (ibit >= ip1 && i2rev > ibit) {
	i2rev -= ibit;
	ibit >>= 1;
      }
      i2rev += ibit;
    }
    ifp1=ip1;                    // Here begins the Danielson-Lanczos section of the routine. 
    while (ifp1 < ip2) {
      ifp2=ifp1 << 1;
      theta=isign*6.28318530717959/(ifp2/ip1);   // Initialize for the trig. recurrence.
      wtemp=sin(0.5*theta);
      wp=-2.0*wtemp*wtemp + I * sin(theta);
      w=1.0 + I * 0.0;
      for (i3=1;i3<=ifp1;i3+=ip1) {
	for (i1=i3;i1<=i3+ip1-2;i1+=2) {
	  for (i2=i1;i2<=ip3;i2+=ifp2) {
	    k1=i2;                                       // Danielson-Lanczos formula:
	    k2=k1+ifp1;
	    h1=k1>>1;
	    h2=k2>>1;
	    for(jd=0;jd<fdeg;jd++){
	      ctemp=w*cdata[h2*fdeg+jd];
	      cdata[h2*fdeg+jd]=cdata[h1*fdeg+jd]-ctemp;
	      cdata[h1*fdeg+jd] += ctemp;
	    }
	  }
	}
	w+=w*wp;                     // Trigonometric recurrence.
      }
      ifp1=ifp2;
    }
    nprev *= n;
  }
}
