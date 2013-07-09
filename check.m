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
