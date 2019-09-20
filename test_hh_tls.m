%

pm = [];
%pm.prog_path = '../bin/gen_neu';
%pm.neuron_model = 'HH-GH';
pm.neuron_model = 'LIF-GH';
pm.simu_method = 'SSC';
%pm.net  = [0 1; 0 0];
pm.net  = [0 1 0; 0 0 1; 0 0 0];
pm.nI   = 0;
pm.scee_mV = 0.5;
pm.scie_mV = 0.0;     % default: 0. Strength from Ex. to In.
pm.scei_mV = 0.5;
pm.scii_mV = 0.0;
pm.pr      = 1.6;
pm.ps_mV   = 0.4;
pm.t    = 1e7;
pm.dt   = 2^-5;
pm.stv  = 0.5;
pm.seed = 'auto';
pm.extra_cmd = '-v';
[X, ISI, ras] = gen_neu(pm);
X = X([1 3], :);
ras(ras(:,1)==2, :) = [];
ras(ras(:,1)==3, 1) = 2;

%randn('state', 2324);
%X = X + 0.01*randn(size(X));

[p, len] = size(X);

od = 10;

tic
[gc, de] = nGrangerTfast(X, od);
toc
gc

%SNR = pm.t/pm.stv;
%SNR = 0.00022421 / 0.01^2;
%SNR = 0.00012354 / 0.01^2;
%SNR = 0.000088840 / 0.01^2;
%SNR = 0.000079850 / 0.01^2;
%SNR = 0.0000057193 / 0.01^2;  % for HH
%SNR = 0.000007 / 0.01^2;    %
SNR = 0.1;

tic
a_v  = zeros(od, p);
a_st = zeros(od*(p-1), p);
for ii = 1:p
  [a, a_vst] = get_subthresidual(X, ras, pm, ii, od, false, SNR);
  a_v(:, ii)  = a_vst(1:od);
  a_st(:, ii) = a_vst(od+2:end);
end
toc % len=2e7, od=10, OLS: 113.867 seconds;  TLS: 164.268 seconds.

figure(25);
plot(a_st, '-o');
ylabel('st coef');
legend('2->1', '1->2');

%[res, a] = get_subthresidual(X, ras, pm, ii, od, true);

