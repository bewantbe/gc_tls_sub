%
% ii: "to" index

function [res, a, a_vst] = get_subthresidual(X, ras, pm, ii, od, is_tls)

%t_exclude_bound = [-5, 12];  % [-5, 12] for HH model
t_exclude_bound = [-1, 3];   % [-1, 3] for HH model
len_short_drop = od + 1;
spike_list = ras(ras(:,1)==ii, 2);
x = X(ii, :)';

[p, len] = size(X);
ST = SpikeTrains(ras, p, len, pm.stv, 0);
ST(ii, :) = [];  % remove itself
ST = ST';

% Methods to assemble non-spiking data into a form for OLS.
switch 1
  case 1    % method 1
    [lo, hi] = ...
      GetQuietPosition(spike_list/pm.stv, pm.t/pm.stv, t_exclude_bound/pm.stv, len_short_drop);
    niv = hi - lo - od + 1;                % length of each good segment
    B   = zeros(sum(niv), 1);              % RHS (v)
    Z   = zeros(sum(niv), od);             % LHS of v
    ZST = zeros(sum(niv), (p-1)*od);       % LHS of spike trains
    tid = 1;
    for k = 1 : length(hi)
      B(tid:tid+niv(k)-1) = x(hi(k) - niv(k) + 1:hi(k))';
      for m = 1 : od
        Z(tid:tid+niv(k)-1, od-m+1) = x(lo(k)+m-1:lo(k)+m-1+niv(k)-1);
        ZST(tid:tid+niv(k)-1, (od-m)*(p-1)+1:(od-m+1)*(p-1)) = ...
          ST(lo(k)+m:lo(k)+m+niv(k)-1, :);
%        ZST(tid:tid+niv(k)-1, (od-m)*(p-1)+1:(od-m+1)*(p-1)) = ...
%          ST(lo(k)+m-1:lo(k)+m-1+niv(k)-1, :);
      end
      tid += niv(k);
    end
    %[B Z](tid-10:tid,:)

  %  B0 = B;
  %  Z0 = Z;
  case 2    % method 2
    len_padding = od;
    [xe, ~, pieces_position] = ...
      RemoveVoltAtSpikeV3(x, len_padding, len_short_drop, spike_list/pm.stv, t_exclude_bound/pm.stv);
    Z = zeros(length(xe) - len_padding, od);
    for m = 1 : od
      Z(:, m) = xe(od-m+1:end-m);
    end
    B = xe(od+1:end);
    id_del = (1-2*len_padding:0)' + pieces_position(2:end)';
    id_del(id_del>length(B)) = [];
    B(id_del) = [];
    Z(id_del(:), :) = [];

  %  assert(norm(Z-Z0) == 0)
  %  assert(norm(B-B0) == 0)
end

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
  case 2
    % smaller fact, closer to OLS.
    % emprically, tune err_drive_ratio so that generator_noise_coef is small
    % because we beleve that the system is smooth.
    %err_drive_ratio = merr^2/mdrive^2;
    err_drive_ratio = 1 / pm.t;
    fact = sqrt(err_drive_ratio / (1 + err_drive_ratio));
    [U, S, V] = svd([fact*B ZE], 'econ');
    a = -V(2:end,end)/(V(1,end)*fact);
%    measurement_noise_r1 = U(:,end) * S(end,end) * V(:,end)';
%    measurement_noise = [1/fact ones(1, size(ZE,2))] .* measurement_noise_r1;
%    res = measurement_noise(:,1);
    res = U(:,end) * S(end,end) / fact;

    measurement_noise_coef = S(end,end)/sqrt(m)
    generator_noise_coef = sqrt(var(measurement_noise(:,1)) - measurement_noise_coef.^2)  # a rude approximation

%    figure(100)
%    semilogy(diag(S), '-o')

%    %figure(101); imshow(V); caxis([-1 1])
%    figure(101); imshow(abs(V));

    a_vst = [];  % TODO
end

ii
var(B)
var(res)

end
