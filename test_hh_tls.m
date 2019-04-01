%

pm = [];
%pm.prog_path = '../bin/gen_neu';
%pm.neuron_model = 'HH-GH';
pm.neuron_model = 'LIF-GH';
pm.simu_method = 'SSC';
pm.net  = [0 1; 0 0];
pm.nI   = 0;
pm.scee_mV = 0.5;
pm.scie_mV = 0.0;     % default: 0. Strength from Ex. to In.
pm.scei_mV = 0.5;
pm.scii_mV = 0.0;
pm.pr      = 1.6;
pm.ps_mV   = 0.4;
pm.t    = 1e4;
pm.dt   = 2^-5;
pm.stv  = 0.5;
pm.seed = 'auto';
pm.extra_cmd = '-v';
[X, ISI, ras] = gen_neu(pm);

X = X + 0.01*randn(size(X));

[p, len] = size(X);

od = 10;

[gc, de] = nGrangerTfast(X, od);
gc

a_v  = zeros(od, p);
a_st = zeros(od*(p-1), p);
for ii = 1:p
  [res, a, a_vst] = get_subthresidual(X, ras, pm, ii, od, false);
  a_v(:, ii)  = a_vst(1:od);
  a_st(:, ii) = a_vst(od+2:end);
end

figure(11);
plot(a_st, '-o');
ylabel('st coef');
legend('2->1', '1->2');

%[res, a] = get_subthresidual(X, ras, pm, ii, od, true);

