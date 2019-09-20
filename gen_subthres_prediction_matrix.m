% Get matrix of the form
% [x(t) x(t-1) ... x(t-od), st(t) st(t-1) ... st(t-od+1)]
%   B   -------- Z -------  ----------- ZST ------------
% x  = X (ii, :)'
% st = ST([1 .. ii-1 ii+1 .. n], :)'

function [B, Z, ZST] = gen_subthres_prediction_matrix(X, ras, pm, ii, od, is_tls)

%t_exclude_bound = [-5, 12];  % [-5, 12] for HH model
t_exclude_bound = [-1, 3];   % [-1, 3] for HH model
len_short_drop = od + 1;
spike_list = ras(ras(:,1)==ii, 2);
x = X(ii, :)';

[p, len] = size(X);
ST = SpikeTrains(ras, p, len, pm.stv, 0);
ST(ii, :) = [];  % remove itself
ST = ST';

% Methods to assemble non-spiking data into a form for OLS ot TLS.
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
  case 2    % method 2, slow, not complete
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

