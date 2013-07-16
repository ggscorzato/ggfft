function [rm] = check(LL,deg,di,par)

% mpirun -np 1 ./test_fft 1 <dim> <L0> ... <L_{dim-1}>  <ones(1,dim)>  dim-1:0 
% LL=[L(dim-1),...L1,L0], with L0= the slowest;

load data_in_ref;
dir=data_in_ref(di:deg:end,1)+i*data_in_ref(di:deg:end,2);
rdir=reshape(dir,LL);
f=fftn(rdir);

load data_out_ref;
dor=data_out_ref(di:deg:end,1)+i*data_out_ref(di:deg:end,2);
rdor=reshape(dor,LL);

r=rdor-f
rm=max(abs(reshape(r,prod(LL),1)));

if (par==1)
 load data_out_check
 doc=data_out_check(di:deg:end,1)+i*data_out_check(di:deg:end,2);
 rdoc=reshape(doc,L2,L1,L0);
 rdc-f 
end
