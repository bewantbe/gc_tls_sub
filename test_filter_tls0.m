% try recover filter coefficients from noisy data
% demo OLS can fail in large noise case

m = 1e6;
n = 10;

merr = 1;

old = randn('state'); randn('state', 34232344);
a = 0.1*randn(n, 1);
randn('state', old);

wb = randn(m+2*n, 1);
x = filter(1, [1; -a], wb);
x = x(n+1:end);
y = x + merr * randn(size(x,1),1);

Z = zeros(m,n);
for k = 1:n
  Z(:,k) = y(n-k+1:end-k);
end
B = y(n+1:end);

fact = 0.705;  % smaller to get closer to OLS, 0.705 for merr=1
[U,S,V] = svd([fact*B Z], 'econ');
fact_all = [fact ones(1,size(Z, 2))];
measurement_noise_r1 = U(:,end) * S(end,end) * V(:,end)';
measurement_noise = 1 ./ fact_all .* measurement_noise_r1;

figure(100);
semilogy(diag(S), '-o');

figure(101);
imshow(abs(V));

figure(102);
plot(std(measurement_noise(:,2:end)), '-o')
ylabel('measurement noise');
std(measurement_noise(:,1))

a
a_est = Z \ B   % correct for merr = 0
a_tls = -V(2:end, end) .* fact_all(2:end) / (V(1,end) * fact)

figure(200);
plot(a, a_est, '-o', a, a_tls, '-o', a, a, '-');
axis([-1 1 -1 1]*0.15)
legend('OLS', 'TLS', 'ans');
legend('location', 'northwest');

