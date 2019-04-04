%

plane = orth(rand(3,2));
plane_normal = null(plane')

m = 1000;
X = randn(m,2) * plane';
merr = 1e-1;
C = [1, 1, 0.5];
Xerr = merr * randn(m, 3) .* C;
X = X + Xerr;

disp('gTLS');
[b, Sigma, eta2] = gTLS(X, C.^2);
b
std_err = diag(chol(Sigma))' / merr ./ C

figure(33);
scatter3(X(:,1),X(:,2),X(:,3));

disp('primitive SVD');
[U, S, V] = svd(X, 'econ');

v_last = V(:,end)

e1 = Xerr*plane_normal;
e2 = U(:,end)*S(end,end);

% std(e1)/merr ~ 1
% std(e2)/merr ~ 1
% S(end,end)/sqrt(m) / merr ~ 1

S(end,end)/sqrt(m) / merr

if (v_last(1)*plane_normal(1)<0)
  v_last = -v_last;
end
if (b(1)*plane_normal(1)<0)
  b = -b;
end

figure(3);
plot(plane_normal, plane_normal, '-+', plane_normal, v_last, '-o', plane_normal, b, '-o');
legend('true', 'v\_last', 'b');
angle_gtls__ = acos(plane_normal' * b)         % should be smaller
angle_v_last = acos(plane_normal' * v_last)

figure(3525)
plot(e1, e2, '-o')
