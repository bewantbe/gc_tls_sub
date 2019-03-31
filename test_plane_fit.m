%

plane = orth(rand(3,2));
plane_normal = null(plane')

m = 1000;
X = randn(m,2) * plane';
merr = 1e-5;
Xerr = merr * randn(m, 3);
X = X + Xerr;

figure(33);
scatter3(X(:,1),X(:,2),X(:,3));

[U, S, V] = svd(X, 'econ');

v_last = V(:,end)

e1 = Xerr*plane_normal;
e2 = U(:,end)*S(end,end);

% std(e1)/merr ~ 1
% std(e2)/merr ~ 1
% S(end,end)/sqrt(m) / merr ~ 1

S(end,end)/sqrt(m) / merr

figure(3525)
plot(e1, e2, '-o')
