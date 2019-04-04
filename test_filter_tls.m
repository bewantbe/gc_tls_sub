% try recover filter coefficients from noisy data
% demo OLS can fail in large noise case

m = 1e5;
n = 10;

merr = 0.6;

old = randn('state'); randn('state', 34232344);
a = 0.1*randn(n, 1);
randn('state', old);

mdrive = 1.0;
wb = mdrive * randn(m+2*n, 1);
b_true = [1; -a];
x = filter(1, b_true, wb);
x = x(n+1:end);
y = x + merr * randn(size(x,1),1);

Z = zeros(m,n);
for k = 1:n
  Z(:,k) = y(n-k+1:end-k);
end
B = y(n+1:end);

% smaller to get closer to OLS
% merr   fact
% 5.0    0.978  % not accurate
% 2.0    0.895
% 1.0    0.707
% 0.6    0.517
% 0.5    0.448
% 0.2    0.194
% 0.1    0.095
% 0.05   0.049

%fact = merr/sqrt(mdrive^2+merr^2);  % merr/sqrt(1+merr*merr)
err_drive_ratio = merr^2/mdrive^2;
fact = sqrt(err_drive_ratio / (1 + err_drive_ratio));
[U,S,V] = svd([fact*B Z], 'econ');
measurement_noise_r1 = U(:,end) * S(end,end) * V(:,end)';
measurement_noise = [1/fact ones(1,size(Z, 2))] .* measurement_noise_r1;

measurement_noise_coef = S(end,end)/sqrt(m)

% norm( ([fact*B Z] - measurement_noise_r1) * V(:, end) ) == 0
%norm( sum([(B - U(:,end)*S(end,end)*V(1,end)/fact)*V(1, end)*fact (Z-U(:,end)*S(end,end)*V(2:end,end)')*V(2:end, end)], 2))

figure(100);
semilogy(diag(S), '-o');

figure(101);
imshow(abs(V));

figure(102);
plot(std(measurement_noise(:,2:end)), '-o')
ylabel('measurement noise');
generator_noise_coef = sqrt(var(measurement_noise(:,1)) - measurement_noise_coef.^2)

a
a_est = Z \ B   % correct for merr = 0
a_tls = -V(2:end, end) / (V(1,end) * fact)

coef_slope = polyfit(a, a_tls, [1 0]>0)

figure(200);
plot(a, a_est, '-o', a, a_tls, '-o', a, a, '-');
axis([-1 1 -1 1]*0.15)
legend('OLS', 'TLS', 'ans');
legend('location', 'northwest');

if 0
  disp('verify gTLS:');
  Sigma = [];
  SNR = (mdrive/merr)^2;
  [b, Sigma, eta2] = gTLS([B Z], Sigma, SNR);
  diag_sqSigma = sqrt(diag(Sigma))'
  eta2
  bb = [b_true b [1; -a_tls]]
end

