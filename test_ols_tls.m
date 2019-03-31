%
randn('state', 34234);
rand('state', 334);

m = 1e5;
n = 10;

a = rand(n, 1);

X = randn(m, n);
Y = X * a + randn(m, 1);

a
est_a = X \ Y

fact = 0.1;  % fact->0, TLS -> OLS
[u, s, v] = svd([fact*Y X], 'econ');
tls_a = -v(2:end, end) / (v(1, end)*fact)

figure(10)
plot(1:n, a, 1:n, est_a, 1:n, tls_a);
legend('true', 'OLS', 'TLS');

figure(11)
plot(a, tls_a, '-o');
legend('TLS');

xlabel('true');
ylabel('a');
legend('location', 'northwest');

figure(13);
semilogy(diag(s), '-o');
