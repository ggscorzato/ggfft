#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <complex.h>
#include <math.h>
#include <mpi.h>
#include "ranlxd.h"
#include "ggfft.h"
#include "ggfft_int.h"  // needed only to test transpose

/* How-to test_fft:
make test_fft
mpirun -np <nprocs> ./test_fft <new> <dim> <deg> <l0> <l1> <l2> ...<n0> <n1> <n2> ... <ix0> <ix1> <ix2> ...
e.g.
mpirun -np 1  ./test_fft 1 2 7 20 20 1 1 1 0
mpirun -np 10 ./test_fft 0 2 7 20 20 5 2 1 0
*/

int main(int argc, char **argv){

  int flag,deg,fsize,j,mu,dim,*fl,*np,*ixor,c0,new,nprocs,*fbasis,*coord,*gcoord,*pcoord,jj,pid,*pi,*po;
  _Complex double *data,*fdata;
  double *rdata,dvar,dvai;
  FILE * fid=NULL;
  char var[128], vai[128], temp[128], filename[50];
  gg_plan testplan;

  /***** Init *****/
  MPI_Init(&argc,&argv);
  MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
  
  /* get from input */
  new=atoi(argv[1]);
  dim=atoi(argv[2]);
  deg=atoi(argv[3]);
  c0=4;
  fl=calloc(dim,sizeof(int));
  np=calloc(dim,sizeof(int));
  ixor=calloc(dim,sizeof(int));
  
  for(mu=0;mu<dim;mu++) fl[mu]=atoi(argv[mu+c0]);
  fsize=1;
  for(mu=0;mu<dim;mu++) fsize*=fl[mu];
  for(mu=0;mu<dim;mu++) np[mu]=atoi(argv[mu+c0+dim]);
  if(new<1){
    for(mu=0;mu<dim;mu++) ixor[mu]=atoi(argv[mu+c0+2*dim]);
  } else if (new==1){
    for(mu=0;mu<dim;mu++) ixor[mu]=dim-mu-1; /* if I generate a new vector I impose ixor=dim-1...0*/
  }

  if(new==-1){
    pi=calloc(dim,sizeof(int));
    po=calloc(dim,sizeof(int));
    for(mu=0;mu<dim;mu++) pi[mu]=atoi(argv[mu+c0+3*dim]);
    for(mu=0;mu<dim;mu++) po[mu]=atoi(argv[mu+c0+4*dim]);
  }
  
  /* init fft */
  gg_init_plan(&testplan,dim,fl,np,ixor,deg,MPI_COMM_WORLD);
  
  /* other allocations */
  fbasis=calloc(dim,sizeof(int));
  coord=calloc(dim,sizeof(int));
  gcoord=calloc(dim,sizeof(int));
  pcoord=calloc(dim,sizeof(int));
  data=calloc(deg*testplan.volproc,sizeof(_Complex double));
  rdata=calloc(deg*testplan.volproc,sizeof(double));
  fdata=calloc(deg*fsize,sizeof(_Complex double));
  
  /***** either we generate a new vector or we take one from disk *****/

  if(new==1){             /*** If I need a new vector, I call random number gen, and print it ***/
    if(nprocs!=1){ 
      fprintf(stdout, "ERROR: initialize test on a single processor. nprocs=%d\n",nprocs);
      exit(-1);
    }
    gauss_vectord(rdata,fsize*deg);
    for(j=0;j<fsize*deg;j++) data[j]=(_Complex double) rdata[j];
    gauss_vectord(rdata,fsize*deg);
    for(j=0;j<fsize*deg;j++) data[j]+= I * rdata[j];
    
    fid = fopen("data_in_ref", "w");
    for(j=0;j<fsize*deg;j++) fprintf(fid,"%15.14e, %15.14e\n",creal(data[j]),cimag(data[j]));
    fclose(fid);
    
  } else if(new<1){            /*** otherwise I read it... ***/
    if(testplan.proc_id==0) {
      fid = fopen("data_in_ref", "r");
      j=0;
      while (fgets(temp, 127, fid) != NULL) {
	sscanf(temp,"%[^','],%s\n",var,vai);
	dvar=atof(var);
	dvai=atof(vai);	
	fdata[j]=dvar + I* dvai;
	j++;
      }
      fclose(fid);
      fid = fopen("data_in_check", "w");
      for(j=0;j<fsize*deg;j++) fprintf(fid,"%15.14e, %15.14e\n",creal(fdata[j]),cimag(fdata[j]));
      fclose(fid);
    }

    /* ... and distribute it ... */
    MPI_Bcast(fdata, fsize*deg, MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);

    /* ... and put the right data in the right place */
    fbasis[dim-1]=1;
    for(mu=dim-2;mu>=0;mu--) fbasis[mu]=fbasis[mu+1]*fl[mu+1];
    for(j=0;j<testplan.volproc*deg;j++){
      jj=0;
      for(mu=0;mu<dim;mu++){
	coord[mu]= (j / testplan.cbasis[mu]) % testplan.lengths[mu];
	gcoord[mu]=coord[mu] + testplan.proc_coords[mu]*testplan.lengths[mu];
	jj+=gcoord[mu]*fbasis[mu];
      }
      data[j]=fdata[jj];
    }
  }                               /*** end of if(new==...) ***/

  if(testplan.proc_id==1) 
    fprintf(stdout,"DEBUG test -- data generated or read, e.g. data[0].re=%g\n",creal(data[0]));

  /***** Actual FFT *****/
  flag=-1;

  if(new==-1){
    transpose(&testplan,data,pi,po);
  } else{
    gg_distributed_multidim_fft(&testplan, flag, data);
  }

  if(testplan.proc_id==1) fprintf(stdout,"DEBUG test -- FFT performed , e.g. data[0].re=%g\n",creal(data[0]));

  /***** write the results.  *****/

  MPI_Barrier(MPI_COMM_WORLD);

  if(new==1){
    sprintf(filename,"data_out_ref");
    fid=fopen(filename,"w");
    for(j=0;j<testplan.volproc*deg;j++) fprintf(fid,"%15.14e, %15.14e\n",creal(data[j]),cimag(data[j]));
    fclose(fid);
  }
  if(new<1){
    sprintf(filename,"data_out_check");
    for(jj=0;jj<deg*testplan.volproc*testplan.totproc;jj++){
      pid=0;
      for(mu=0;mu<dim;mu++){
        gcoord[mu]= (jj / fbasis[mu] ) % fl[mu];
        coord[mu] = gcoord[mu]%testplan.lengths[mu];
        pcoord[mu] = gcoord[mu] / testplan.lengths[mu];
        pid+=pcoord[mu]*testplan.nbasis[mu];
      }

      MPI_Barrier(MPI_COMM_WORLD);
      if(pid==testplan.proc_id){
        j=0;
	for(mu=0;mu<testplan.dim;mu++) j+=coord[mu]*testplan.cbasis[mu];

	fid=fopen(filename,"a");
        fprintf(fid,"%15.14e, %15.14e\n",creal(data[j]),cimag(data[j]));fflush(fid);
	fclose(fid);
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }
  }
  if(testplan.proc_id==0) fprintf(stdout,"DEBUG test -- data written\n");
  
  /***** Finalize *****/
  gg_destroy_plan(&testplan);
  MPI_Finalize();
  return(0);

}
