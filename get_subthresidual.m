%

function [res, a] = get_subthresidual(X, ras, pm, ii, od, is_tls)

ii = 1;  % "to" index
t_exclude_bound = [-5, 12];  % [-5, 12] for HH model
len_short_drop = od + 1;
spike_list = ras(ras(:,1)==ii, 2);
x = X(ii, :)';

% Methods to assemble non-spiking data into a form for OLS.
switch 1
  case 1
    tic()
    % method 1
    [lo, hi] = ...
      GetQuietPosition(spike_list/pm.stv, pm.t/pm.stv, t_exclude_bound/pm.stv, len_short_drop);
    niv = hi - lo - od + 1;
    B = zeros(sum(niv), 1);
    Z = zeros(sum(niv), od);
    tid = 1;
    for k = 1 : length(hi)
      %RHS
      B(tid:tid+niv(k)-1) = x(hi(k) - niv(k) + 1:hi(k))';
      %LHS
      for m = 1 : od
        Z(tid:tid+niv(k)-1, od-m+1) = x(lo(k)+m-1:lo(k)+m-1+niv(k)-1);
      end
      tid += niv(k);
    end
    toc()
    %[B Z](tid-10:tid,:)

  %  B0 = B;
  %  Z0 = Z;
  case 2
    tic()
    % method 2
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
    toc()

  %  assert(norm(Z-Z0) == 0)
  %  assert(norm(B-B0) == 0)
end

%var(f_res(Z, B)) / var(B)
%var(f_res1(Z, B)) / var(B)
%de(1,1) / var(X(1,:))

ZE = [Z ones(size(Z,1),1)];

% methods to "driving noise"
switch (is_tls>0) + 1
  case 1
    a = ZE \ B;
    res = B - ZE*a;
  case 2
    % smaller fact, closer to OLS.
    %err_drive_ratio = merr^2/mdrive^2;
    err_drive_ratio = 1 / pm.t;   % 
                              % emprically, tune it so that generator_noise_coef~0
                              % because we beleve that the system is smooth.
    fact = sqrt(err_drive_ratio / (1 + err_drive_ratio));
    [U, S, V] = svd([fact*B ZE], 'econ');
    a = -V(2:end,end)/(V(1,end)*fact);
%    measurement_noise_r1 = U(:,end) * S(end,end) * V(:,end)';
%    measurement_noise = [1/fact ones(1, size(ZE,2))] .* measurement_noise_r1;
%    res = measurement_noise(:,1);
    res = U(:,end) * S(end,end) / fact;

    measurement_noise_coef = S(end,end)/sqrt(m)
    generator_noise_coef = sqrt(var(measurement_noise(:,1)) - measurement_noise_coef.^2)

%    figure(100)
%    semilogy(diag(S), '-o')

%    %figure(101); imshow(V); caxis([-1 1])
%    figure(101); imshow(abs(V));

end

end
