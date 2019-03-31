%

pm = [];
%pm.prog_path = '../bin/gen_neu';
pm.neuron_model = 'HH-GH';
pm.simu_method = 'SSC';
pm.net  = [0 1; 0 0];
pm.nI   = 0;
pm.scee_mV = 0.5;
pm.scie_mV = 0.0;     % default: 0. Strength from Ex. to In.
pm.scei_mV = 0.0;
pm.scii_mV = 0.0;
pm.pr      = 1.6;
pm.ps_mV   = 0.4;
pm.t    = 1e7;
pm.dt   = 2^-5;
pm.stv  = 0.5;
pm.seed = 'auto';
pm.extra_cmd = '-v';
[X, ISI, ras] = gen_neu(pm);

%X = X + 0.0*randn(size(X));

[p, len] = size(X);
s_t = (1:len)*pm.stv;

od = 10;

tic()
[gc, de] = nGrangerTfast(X, od);
gc
toc()

f_res =  @(X, Y) Y - X * (X \ Y);  % E: X*a = Y + E
f_res1 =  @(X, Y) Y - [ones(size(X,1),1) X] * ([ones(size(X,1),1) X] \ Y);  % E: [1 X]*a = Y + E

[res, a] = get_subthresidual(X, ras, pm, ii, od, false);

var(B)
var(res)

%[res, a] = get_subthresidual(X, ras, pm, ii, od, true);
%var(res)

%figure(10);
%plot(s_t, X);
%line([1; 1] .* lo(:)'*pm.stv, [-20; 50]*ones(1, length(lo)), 'color', [1 1 0]);
%line([1; 1] .* hi(:)'*pm.stv, [-20; 50]*ones(1, length(lo)), 'color', [0 1 1]);

