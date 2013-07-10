%%%%%%%%%%%%%%%%%%%% check with matlab fft
% matlab check only of the serial version. The parallel version is checked versus the serial one.
% this is not a matlab code. It is just a help.

load vector.in
load vector_part_0.out
deg=3;
dim=[16,4];

d=1..deg;

v${d}=reshape(vector(d:deg:end,1) + i*vector(d:deg:end,2),dim);
wc${d}=reshape(vector_part_0(d:deg:end,1) + i*vector_part_0(d:deg:end,2),dim);

wm${d}=fftn(v${d});

max(abs(wm${d}-wc${d}))


%%%%%%%%%%%%%%%%%%%% check between the parallel and serial (ixor=[dim-1...0] only)

mpirun -np 1 ./test 1 <dim> <deg> <full length dir 0> <full length dir 1> ... 1 1 [dim times] ...dim-1 ...0 
mpirun -np <np> ./test 0 <dim> <deg> <full length dir 0> <full length dir 1> ...<# procs dir 0> <# procs dir 1> ...dim-1 ...0 

diff data_in_ref data_in_check    #should coincide up to rounding errors
diff data_out_ref data_out_check    #should coincide up to rounding errors
