#include <stdlib.h>
#include <stdio.h>
#include <complex.h>
#include <math.h>
#include <mpi.h>
#include "ggfft.h"
#include "ggfft_int.h"

#define DEBUG 0
/*****
      Initialization routine. It should be called after MPI_Init, but before any function in ggfft.
      dim: dimension of the data to be FFT,
      flengths[]: array of length dim;  it contains the full lengths of points foreach direction.
      nprocl[]: array of length dim; it contains the number of processors foreach direction.
      ixor[]: array of length dim; ixor[0] is the fastest direction, ixor[dim-1] the slowest in lexicografic order.
      the three pointers above should be already allocated before passing them to gg_init.
*****/
void gg_init_plan(gg_plan *pln, int dim, int *flengths, int *nprocl, int *ixor, int deg){

  int mu,i,*periods,*isfft,check,maxprocdir;
  MPI_Comm cart_comm;

  /*** copy the user set variables to global variables ***/
  pln->dim=dim;
  pln->flengths=flengths;
  pln->nprocl=nprocl;
  pln->ixor=ixor;

  /*** define coordinates of this process w.r.t. the whole lattice, both sequential (pln->proc_id) and vector
     (pln->proc_coords) ***/
  pln->proc_coords=calloc(pln->dim,sizeof(int));
  periods=calloc(pln->dim,sizeof(int));
  for(mu=0;mu<pln->dim;mu++) periods[mu]=1; // torus
  MPI_Cart_create(MPI_COMM_WORLD, pln->dim, pln->nprocl, periods, 1, &cart_comm);
  MPI_Comm_rank(cart_comm,&pln->proc_id);
  MPI_Cart_coords(cart_comm,pln->proc_id,pln->dim,pln->proc_coords);
  MPI_Comm_size(cart_comm,&check);
  pln->totproc=1;
  for(mu=0;mu<pln->dim;mu++) pln->totproc*=pln->nprocl[mu];
  if(check!=pln->totproc){
    if(pln->proc_id==0){ 
      fprintf(stderr,"ERROR: the number or processes defined by the lattice (%d) does not match with the one" 
	      "set by mpirun(%d).\n",pln->totproc,check);
    }
    exit(-1);
  }

  /* basis for the conversion of indices between sequential and vector. pln->nbasis should be the one used by
     MPI_Cart_coords. This assumes the one implemented in openmpi */
  pln->nbasis=calloc(pln->dim,sizeof(int)); 
  pln->nbasis[pln->dim-1]=1;  
  for(mu=pln->dim-2; mu>=0;mu--) pln->nbasis[mu]=pln->nbasis[mu+1]*pln->nprocl[mu+1];
  check=0;
  for(mu=0;mu<pln->dim;mu++) check+=pln->nbasis[mu]*pln->proc_coords[mu];
  if(check!=pln->proc_id){
    fprintf(stderr,"ERROR: processors mapping error in proc: %d. It does not match with: %d\n",pln->proc_id,check);
    exit(-1);
  }
  
  /*** define local coordinates ***/
  pln->lengths=calloc(pln->dim,sizeof(int));
  pln->volproc=1;
  for(mu=0;mu<pln->dim;mu++){
    pln->lengths[mu]=pln->flengths[mu]/pln->nprocl[mu]; // local lengths of the portion of lattice in a process.
    pln->volproc*=pln->lengths[mu]; // local volume
  }

  pln->cbasis=calloc(pln->dim,sizeof(int));
  pln->cbasis[pln->ixor[0]]=1;
  for(mu=1;mu<pln->dim;mu++) pln->cbasis[pln->ixor[mu]]=pln->cbasis[pln->ixor[mu-1]]*pln->lengths[pln->ixor[mu-1]];

  /*** determine the directions and the dimension of the serial FFT's ***/
  pln->fftdir=calloc(pln->dim,sizeof(int));
  pln->nfftdir=calloc(pln->dim,sizeof(int));
  isfft=calloc(pln->dim,sizeof(int));
  for(mu=0;mu<pln->dim;mu++) pln->fftdir[mu]=-1;
  for(mu=0;mu<pln->dim;mu++) pln->nfftdir[mu]=-1;
  for(mu=0;mu<pln->dim;mu++) isfft[mu]=0;
  pln->fftdim=0;
  mu=0;
  while(pln->nprocl[pln->ixor[mu]]>1) mu++;
  pln->fftdir[0]=pln->ixor[mu];
  isfft[pln->ixor[mu]]=1;
  pln->fftdim=1;
  if(dim>mu+1){
    if(pln->nprocl[pln->ixor[mu+1]]==1 &&  pln->dim%2==0){
      pln->fftdir[1]=pln->ixor[mu+1];
      isfft[pln->ixor[mu+1]]=1;
      pln->fftdim=2;
    }
  }
  if (pln->dim>mu+2){
    if(pln->nprocl[pln->ixor[mu+1]]==1 && pln->nprocl[pln->ixor[mu+2]]==1 && pln->dim%3==0){
      pln->fftdir[1]=pln->ixor[mu+1];
      isfft[pln->ixor[mu+1]]=1;
      pln->fftdir[2]=pln->ixor[mu+2];
      isfft[pln->ixor[mu+2]]=1;
      pln->fftdim=3;
    }
  }
  if(pln->dim>mu+3){
    if(pln->nprocl[pln->ixor[mu+1]]==1 && pln->nprocl[pln->ixor[mu+2]]==1 && pln->nprocl[pln->ixor[mu+3]]==1 
       && pln->dim%4==0){
      pln->fftdir[1]=pln->ixor[mu+1];
      isfft[pln->ixor[mu+1]]=1;
      pln->fftdir[2]=pln->ixor[mu+2];
      isfft[pln->ixor[mu+2]]=1;
      pln->fftdir[3]=pln->ixor[mu+3];
      isfft[pln->ixor[mu+3]]=1;
      pln->fftdim=4;
    }
  }
  if(pln->fftdim==0){
    if(pln->proc_id==0) fprintf(stderr,"ERROR: at least one direction should not be distributed.\n");
    exit(-1);
  }
  i=0;
  for(mu=0;mu<pln->dim;mu++){
    if(!isfft[mu]){
      pln->nfftdir[i] = mu;
      i++;
    }
  }
  /*** allocate labels and indices for exchange vectors ***/
  pln->kk = calloc(pln->totproc,sizeof(int));
  pln->pp = calloc(pln->totproc,sizeof(int));

  maxprocdir=1;
  for(mu=0;mu<pln->dim;mu++){
    if(pln->nprocl[mu]>maxprocdir) maxprocdir=pln->nprocl[mu];
  }
  pln->label_ = (int*)malloc(maxprocdir*pln->volproc*sizeof(int));
  pln->label = (int**)malloc(maxprocdir*sizeof(int*));
  pln->label[0]=pln->label_;
  for(i=1;i<maxprocdir;i++){
    pln->label[i] = pln->label[i-1]+pln->volproc;
  }
  pln->mail__=(_Complex double*)malloc(maxprocdir*pln->volproc*deg*sizeof(_Complex double));
  pln->mail_=(_Complex double**)malloc(maxprocdir*pln->volproc*sizeof(_Complex double*));
  pln->Mail = (_Complex double***)malloc(maxprocdir*sizeof(_Complex double**));
  pln->mail_[0]=pln->mail__;
  for(i=1;i<maxprocdir*pln->volproc;i++){
    pln->mail_[i]=pln->mail_[i-1] + deg;
  }
  pln->Mail[0]=pln->mail_;
  for(i=1;i<maxprocdir;i++){
    pln->Mail[i]=pln->Mail[i-1] + pln->volproc;
  }

  free(periods);
  free(isfft);
}

/***** function to be called when the plan is not needed anymore, to free memory *****/
void gg_destroy_plan(gg_plan *pln){
  free(pln->lengths);
  free(pln->proc_coords);
  free(pln->nbasis);
  free(pln->cbasis);
  free(pln->fftdir);
  free(pln->nfftdir);
  free(pln->kk);
  free(pln->pp);
  free(pln->label);
  free(pln->Mail);
}

/*****  distributed multidimensional FFT.
	fftflag: 1 direct FFT, -1 inverse FFT
	vv: array of size= deg * (\prod_mu pln->flengths[mu]). This vector will be fft'd.
	deg: internal degeneracy of the data in vv. 
*****/

void gg_distributed_multidim_fft(gg_plan *pln, int fftflag, _Complex double * vv, int deg){

  int nfftpoints,repl_slow,repl_fast,mu,it,jj;
  int *nn, *inperm, *outperm;
  _Complex double * pnt0;

#if DEBUG
  fprintf(stdout,"DEBUG gmfft (%d) -- entered. ",pln->proc_id); fflush(stdout);
  if(pln->proc_id==0){
    fprintf(stdout,"fftdim=%d, pln->fftdir=(",pln->fftdim); 
    for(mu=0;mu<pln->fftdim;mu++){
      fprintf(stdout,"%d,",pln->fftdir[mu]);
    }
    fprintf(stdout,")\n");
  }
#endif

  inperm=calloc(pln->dim,sizeof(int));
  for(mu=0; mu<pln->dim;mu++) inperm[mu]=mu;
  outperm=calloc(pln->dim,sizeof(int));
  for(mu=0; mu<pln->dim;mu++) outperm[mu]=mu;
  nn=calloc(pln->fftdim,sizeof(int));
  /* last index procedes most rapidly in the serial fft, hence: */
  for(mu=0; mu<pln->fftdim;mu++) nn[pln->fftdim-1-mu] = pln->flengths[pln->fftdir[mu]]; 
  for(mu=0;mu<pln->dim;mu++) pln->lengths[mu]=pln->flengths[mu]/pln->nprocl[mu];
  pln->cbasis[pln->ixor[0]]=1;
  for(mu=1;mu<pln->dim;mu++) pln->cbasis[pln->ixor[mu]]=pln->cbasis[pln->ixor[mu-1]]*pln->lengths[pln->ixor[mu-1]];
  repl_fast=pln->cbasis[pln->fftdir[0]];
  nfftpoints=1;
  for(mu=0;mu<pln->fftdim;mu++) nfftpoints*=pln->lengths[pln->fftdir[mu]];
  repl_slow=pln->volproc/(repl_fast*nfftpoints);

  for(it=0;it<pln->dim;it+=pln->fftdim){  /* loop on fft subspaces */

    for(jj=0;jj<repl_slow;jj++){  /* loop on all points in orthogonal directions */
      pnt0=vv+(jj*(nfftpoints)*repl_fast*deg);
      serial_multidim_fft(pnt0,repl_fast*deg,nn,pln->fftdim,fftflag);  /* fft on the LAST pln->fftdim directions */
    }
    
    if(it<pln->dim-pln->fftdim){                   /* transpose after fft, except the last time */

      for(mu=0; mu<pln->fftdim;mu++){
	outperm[pln->nfftdir[it+mu]] = inperm[pln->fftdir[mu]];
	outperm[pln->fftdir[mu]] = inperm[pln->nfftdir[it+mu]];
      }

      transpose(pln,vv,deg,inperm,outperm);
      
      for(mu=0; mu<pln->fftdim;mu++) nn[pln->fftdim-1-mu] = pln->flengths[outperm[pln->fftdir[mu]]];
      for(mu=0;mu<pln->dim;mu++) pln->lengths[mu]=pln->flengths[outperm[mu]]/pln->nprocl[mu];
      pln->cbasis[pln->ixor[0]]=1;
      for(mu=1;mu<pln->dim;mu++){
	pln->cbasis[pln->ixor[mu]]=pln->cbasis[pln->ixor[mu-1]]*pln->lengths[pln->ixor[mu-1]];
      }
      repl_fast=pln->cbasis[pln->fftdir[0]];
      nfftpoints=1;
      for(mu=0;mu<pln->fftdim;mu++) nfftpoints*=pln->lengths[pln->fftdir[mu]];
      repl_slow=pln->volproc/(repl_fast*nfftpoints);
      
      for(mu=0; mu<pln->dim;mu++) inperm[mu]=outperm[mu];

    }

  }  /* endo of loop on fft subspaces */

  for(it=pln->dim-2*pln->fftdim;it>=0;it-=pln->fftdim){/* permute back by the reversed sequence of transposes */
    for(mu=0; mu<pln->fftdim;mu++){
      outperm[pln->nfftdir[it+mu]] = inperm[pln->fftdir[mu]];
      outperm[pln->fftdir[mu]] = inperm[pln->nfftdir[it+mu]];
    }

    transpose(pln,vv,deg,inperm,outperm);
    for(mu=0; mu<pln->dim;mu++) inperm[mu]=outperm[mu];
  }

  /* put back the global variables that we changed */
  for(mu=0;mu<pln->dim;mu++) pln->lengths[mu]=pln->flengths[mu]/pln->nprocl[mu];
  pln->cbasis[pln->ixor[0]]=1;
  for(mu=1;mu<pln->dim;mu++) pln->cbasis[pln->ixor[mu]]=pln->cbasis[pln->ixor[mu-1]]*pln->lengths[pln->ixor[mu-1]];

  free(inperm);
  free(outperm);
  free(nn);
}

/*****
      Transposes the dim-dimensional array vv (linearly arranged), according to the input/output permutations
      permutin/permutout.
      -vv[]: array of size= deg * (\prod_mu pln->flengths[mu]). This vector will be transposed. [I&O]
      -deg: internal degeneracy in the data of vv.
      -permutin[]: permutation of the indices assumed for the input
      -permutout[]: permutation of the indices to be found in the output
*****/

void transpose(gg_plan *pln, _Complex double * vv, int deg, int * permutin, int * permutout){

  int seqin,seqout,mu,ind,i,j,ka,proc_out,nprocdest;
  int *lleng_in,*lleng_out,*temp_orig,*locoIn,* glcoIn,*locoOut,* glcoOut,*proc_coordOut,*cbasisIn,*cbasisOut;

  lleng_in=calloc(pln->dim,sizeof(int));
  lleng_out=calloc(pln->dim,sizeof(int));
  cbasisIn=calloc(pln->dim,sizeof(int));
  cbasisOut=calloc(pln->dim,sizeof(int));
  locoIn=calloc(pln->dim,sizeof(int));
  locoOut=calloc(pln->dim,sizeof(int));
  glcoIn=calloc(pln->dim,sizeof(int));
  glcoOut=calloc(pln->dim,sizeof(int));
  proc_coordOut=calloc(pln->dim,sizeof(int));
  temp_orig=calloc(pln->dim,sizeof(int));

  for(mu=0; mu<pln->dim; mu++){
    lleng_in[mu]=pln->flengths[permutin[mu]]/pln->nprocl[mu];
    lleng_out[mu]=pln->flengths[permutout[mu]]/pln->nprocl[mu];
  }

  cbasisIn[pln->ixor[0]]=1;
  cbasisOut[pln->ixor[0]]=1;
  for(mu=1;mu<pln->dim;mu++){
    cbasisIn[pln->ixor[mu]]=cbasisIn[pln->ixor[mu-1]]*lleng_in[pln->ixor[mu-1]];
    cbasisOut[pln->ixor[mu]]=cbasisOut[pln->ixor[mu-1]]*lleng_out[pln->ixor[mu-1]];
  }

  for(ind=0; ind < pln->totproc; ind++) pln->kk[ind]=0;
  for(ind=0; ind < pln->totproc; ind++) pln->pp[ind]=-1;
  nprocdest=0;

  for(i=0; i<pln->volproc; i++){    // main loop: for every local site

    seqin=i;
    for(mu=0; mu < pln->dim; mu++){   // construct local an global IN index (from the sequential)
      locoIn[mu]=(seqin / cbasisIn[mu]) % lleng_in[mu];
      glcoIn[mu]= locoIn[mu]+lleng_in[mu]*pln->proc_coords[mu];
    }

    // transpose the global indices glcoOut[mu] = glcoIn[po[pi^-1[mu]]]:
    for(mu=0; mu < pln->dim; mu++) temp_orig[permutin[mu]]=glcoIn[mu]; 
    for(mu=0; mu < pln->dim; mu++) glcoOut[mu]=temp_orig[permutout[mu]];
    proc_out=0;
    seqout=0;
    for(mu=0; mu < pln->dim; mu++){     // reconstruct local and sequential OUT index
      proc_coordOut[mu]=glcoOut[mu]/lleng_out[mu];
      locoOut[mu]=glcoOut[mu] % lleng_out[mu];
      proc_out+=proc_coordOut[mu]*pln->nbasis[mu];
      seqout+=locoOut[mu]*cbasisOut[mu];
    }

    if(pln->kk[proc_out]==0){
      pln->pp[proc_out]=nprocdest;
      nprocdest++;
    }
    /* store index and variable for send */
    for(j=0;j<deg;j++) pln->Mail[pln->pp[proc_out]][pln->kk[proc_out]][j]=vv[i*deg+j];
    pln->label[pln->pp[proc_out]][pln->kk[proc_out]]=seqout;
    pln->kk[proc_out]++;
  } // end of main loop on every local site.

  MPI_Barrier(MPI_COMM_WORLD);

  for(ind=0; ind < pln->totproc; ind++){  // Send
    if(ind!=pln->proc_id && pln->kk[ind] != 0){
      MPI_Sendrecv_replace(pln->label[pln->pp[ind]],pln->kk[ind],MPI_INT,ind,pln->totproc+pln->proc_id,
			   ind,pln->totproc+ind,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
      MPI_Sendrecv_replace(pln->Mail[pln->pp[ind]][0],pln->kk[ind]*deg,MPI_DOUBLE_COMPLEX,ind,pln->proc_id,
			   ind,ind,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);

  for(ind=0; ind< pln->totproc; ind++){   // copy back the exchanged data onto the array vv
    for(ka=0; ka< pln->kk[ind]; ka++){
      for(j=0;j<deg;j++){
	vv[pln->label[pln->pp[ind]][ka]*deg+j]=pln->Mail[pln->pp[ind]][ka][j];
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
       in cdata[0...(prod_nn*deg)-1]. The routine replaces cdata by its (sdim)-dimensional discrete Fourier
       transform, if isign is input as 1.  nn[0...(sdim)-1] is an integer array containing the lengths of each
       dimension (number of complex values), which MUST all be powers of 2. data is a real array of length twice
       the product of these lengths, in which the data are stored as in a multidimensional complex array: real and
       imaginary parts of each element are in consecutive locations, and the sdim-1 index of the array increases
       most rapidly (the 0th index more slowly) as one proceeds along data.  For a two-dimensional array, this is
       equivalent to storing the array by rows. If isign is input as -1, data is replaced by its inverse transform
       times the product of the lengths of all dimensions.  call it with isign=-1 to have the same definition of
       matlab fft: F_k = \sum_j=0^N-1 exp[-i 2 \pi j k] f_k
*****/

void serial_multidim_fft(_Complex double * cdata, int fdeg, int * nn, int sdim, int isign)
{
  int id,jd;
  unsigned long i1,i2,i3,i2rev,i3rev,ip1,ip2,ip3,ifp1,ifp2;
  unsigned long ibit,k1,k2,n,nprev,nrem,ntot;
  unsigned long j3,j3rev,h1,h2;
  _Complex double w,wp,ctemp,swap;
  double theta,wtemp;       //Double precision for trigonometric recurrences.
  for (ntot=1,id=0;id<sdim;id++) ntot *= nn[id];   // Compute total number of sites.
  nprev=1;

  for (id=sdim-1;id>=0;id--) {           // Main loop over the dimensions.
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
