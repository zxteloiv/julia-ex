using Distributions

Sigma = [1. 0.; 0. 1.];

mu1 = [1., -1.]; 
x1 = rand(MvNormal(mu1, Sigma), 200);

mu2 = [5.5, -4.5];
x2 = rand(MvNormal(mu2, Sigma), 200);

mu3 = [1., 4.]; 
x3 = rand(MvNormal(mu3, Sigma), 200);

mu4 = [6., 4.5]; 
x4 = rand(MvNormal(mu4, Sigma), 200);

mu5 = [9., 0.0]; 
x5 = rand(MvNormal(mu5, Sigma), 200);

x = [x1 x2 x3 x4 x5]

println("x = $(repr(x))")
