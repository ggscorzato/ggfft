
load data_in_ref

dir=data_in_ref(di:deg:end,1)+i*data_in_ref(di:deg:end,2);
rdir=reshape(dir,LL);
f=fftn(rdir);

f1=fft(rdir,[],3);
f1p=permute(f1,[1,3,2]);
f2=fft(f1p,[],3);
f2p=permute(f2,[3,2,1]);
f3=fft(f2p,[],3);
f3p=permute(f3,[3,2,1]);
f3pp=permute(f3p,[1,3,2]);

f-f3pp % OK
