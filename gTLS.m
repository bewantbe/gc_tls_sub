% TLS with inhomogenous measurement error.
%
% Inputs:
%   Z: m x n matrix, m data points, each is dimension n.
%   Sigma: error matrix.
%
% Form 1: tls_error_mode == 'mu' (default)
%   Assume Z = X + E * C
%          where E is i.i.d measurement error (matrix size the same as X).
%          C is error generator matrix, i.e. C'*var(E)*C is the
%            variance-covariance matrix of errors.
%   Want vector b such that
%     X * b = 0
%   Solved by
%     b = normalize( C \ V(:,n) )
%     Sigma_real = var(E * C) = C' * C * s(n)^2/m / tw_correction;
%     where
%       U*S*V' = Z/C = X/C + E
%       X/C * V(:,n) = 0
%     variance of E (mu^2*I) is estimated as
%       s(n)^2*I = m*var(E) = m*mu^2*I
%   Note: when more than one dimension is small, e.g. s(n-1)==s(n),
%         the solution of b is either non-unique or data too noisy.
%   Verify model (X * b = 0), necessary condition:
%     exam distribution of E * V(:,n) = U(:,n)*s(n).
%     e.g. should be gaussian distributed for measurement.
%          if not, maybe the space is curved (bended).
%
% Form 2: tls_error_mode == 'OLS'
%   Assume Z = X + E * C + [E0*eta, zeros(m,n-1)]
%          where E0 and E are uncorrelated, var(E0) = var(E(:,1))
%          known: SNR = eta^2 / (C(:,1)'*C(:,1)) = eta^2 / Sigma(1,1)
%   Want vector b such that
%     X * b = 0 and b(1) == 1
%     i.e.
%       Fitting OLS
%         Y = X(:, 2:n) * -b(2:n) + E0*eta
%       However, measurement of Y and X(:, 2:n) are noisy
%         Z(:,1)   = Y + E*C(:,1)
%         Z(2:n,1) = X(:, 2:n) + E*C(2:n,1)
%   Solved by
%     C_new'*C_new = C'*C + [eta^2 0; 0 0], then follow Form 1 for C_new
%     b = b/b(1)
%     eta^2      = Sigma(1,1) * SNR/(SNR+1)
%     Sigma(1,1) = Sigma(1,1) * 1/(SNR+1)

function [b, Sigma, eta2] = gTLS(Z, Sigma, SNR)
  ols_mode = exist('SNR', 'var') && ~isempty(SNR);
  [m, n] = size(Z);
  if n <= 1
    b = 0;
    Sigma = var(Z);
  end
  neen_correction = false;
  if m <= n
    warning('Input Z rank deficient.');
  end
  if ~exist('Sigma', 'var') || isempty(Sigma)
    Sigma = eye(n);
  end
  if isvector(Sigma)
    Sigma = diag(Sigma);
  end
  if ols_mode
%    old_Sigma = Sigma;
    Sigma(1,1) = Sigma(1,1) * (1 + SNR);
  end
  C = chol(Sigma);

  [U, S, V] = svd(Z / C, 'econ');
  sval = diag(S);
  b = C \ V(:,n);
  if neen_correction
    % Zhigang Bao, Guangming Pan and Wang Zhou (2012) Tracy-Widom law for the extreme eigenvalues of sample correlation matrices
    tw_correction = ((sqrt(n)-sqrt(m))^2 + 1.49*(sqrt(m)-sqrt(n))*(n^-0.5-m^-0.5)^(1/3))/m;
    % tw_correction = (1-sqrt(n/m))^2 + 1.49/sqrt(m)/n^(1/6) (n<<m)
  else
    tw_correction = 1;
  end
  Sigma = Sigma * sval(n)^2/m / tw_correction;

  if ols_mode
    b = b / b(1);
    eta2 = Sigma(1,1) * SNR/(SNR+1);
    Sigma(1,1) = Sigma(1,1) * 1/(SNR+1);
  else
    b = b / norm(b);
    eta2 = [];
  end
end

%{
% test Tracy-Widom law
ns=1000; m=1e4; n=10; a=zeros(ns,1); for k=1:ns s=svd(randn(m,n)); a(k)=(s(1)^2-(sqrt(m-1)+sqrt(n))^2)/(sqrt(m-1)+sqrt(n))/((m-1)^-0.5 + n^-0.5)^(1/3); end; figure(2); hist(a)
ns=100000; m=1e2; n=10; a=zeros(ns,1); for k=1:ns s=svd(randn(m,n)); a(k)=((s(1)^2-(sqrt(m-1)+sqrt(n))^2)/(sqrt(m-1)+sqrt(n))/((m-1)^-0.5 + n^-0.5)^(1/3)+1.325)/1.22; end; figure(2); hist(a); std(a)

s(1)^2 = (sqrt(m-1)+sqrt(n))^2 + (1.22*x - 1.325)*(sqrt(m-1)+sqrt(n))*((m-1)^-0.5 + n^-0.5)^(1/3)

s(n)^2/m = m/((sqrt(m-1)+sqrt(n))^2 - 1.325*(sqrt(m-1)+sqrt(n))*((m-1)^-0.5 + n^-0.5)^(1/3))

sqrt((sqrt(m-1)+sqrt(n))^2 - 1.325*(sqrt(m-1)+sqrt(n))*((m-1)^-0.5 + n^-0.5)^(1/3))/m

ns=1000; m=1e4; n=10; a=zeros(ns,1); for k=1:ns s=svd(randn(m,n)); a(k)=(s(n)^2 - (sqrt(n)-sqrt(m))^2)/(sqrt(m)-sqrt(n))/(n^-0.5-m^-0.5)^(1/3); end; figure(2); hist(a); std(a)
1.45+-1.31
%}
