%
% ii: "to" index

function [a, a_vst] = get_subthresidual(X, ras, pm, ii, od, is_tls, SNR)

[B, Z, ZST] = gen_subthres_prediction_matrix(X, ras, pm, ii, od, is_tls);

%f_res  = @(X, Y) Y - X * (X \ Y);  % E: X*a = Y + E
%f_res1 = @(X, Y) Y - [ones(size(X,1),1) X] * ([ones(size(X,1),1) X] \ Y);  % E: [1 X]*a = Y + E

%var(f_res(Z, B)) / var(B)
%var(f_res1(Z, B)) / var(B)
%de(1,1) / var(X(1,:))

%figure(10);
%s_t = (1:len)*pm.stv;
%plot(s_t, X);
%line([1; 1] .* lo(:)'*pm.stv, [-20; 50]*ones(1, length(lo)), 'color', [1 1 0]);
%line([1; 1] .* hi(:)'*pm.stv, [-20; 50]*ones(1, length(lo)), 'color', [0 1 1]);

ZE = [Z ones(size(Z,1),1)];

% for debug
%fullZ = [(od+(1:length(B)))'*pm.stv B Z ZST];
%fullZ(round((ras(2,2)-10)/pm.stv)+(1:100), :)
%fullZ(1:100, :)

% methods to "driving noise"
switch (is_tls>0) + 1
  case 1
    a = ZE \ B;
    res = B - ZE*a;
    
    a_vst = [ZE ZST] \ B;
    
    eta2 = var(res);
  case 2
    % higher SNR, closer to OLS.
    % emprically, tune SNR so that eta2 is small
    % because we beleve that the system is smooth.
    Sigma0 = [ones(1, od+1) 1e-12];
    [b_v  , Sigma, eta2] = gTLS([B ZE], Sigma0, SNR);

    Sigma0 = [ones(1, od+1) 1e-14 1e-13*ones(1, od)];
    [b_vst, Sigma, eta2] = gTLS([B ZE ZST], Sigma0, SNR);
    
    a     = -b_v(2:end);
    a_vst = -b_vst(2:end);
    
    Sigma1 = Sigma(1,1)
end

ii
eta2
var(B)

end
