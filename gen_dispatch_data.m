% data for generator dispatch problem
% dual decomposition problem
% https://www.youtube.com/watch?v=Kwli6FkYQYY
%% demand data
D = 2; % number of days
T = D*24*4; % number of 15 minute intervals
t= (1:T)/4; % time, in hours
rand('state',0);
d = 8+ ... % constant demand 
    4*sin(2*pi*t/24) +...  % diurnal variation 
    0.05*t+...  % a linear trend component 
    1.5*rand(1,T); % random demand variation
% generator data
n = 4; % number of generators
Pmax = [6 5 4 2];  % generator capacities
Pmin = [3 1 1 0];  % generator capacities
R = [0.1 0.2 0.5 1.5];  % ramp-rate limits
alpha = [1 1.2 1.5 2]; % linear cost fct coeffs
beta = 0.1*ones(1,n); % quadratic cost fct coeffs
gamma = [3 0.5 0.5 0.1]; % power change cost fct coeffs

Qt = ones(1, T);
step_size = 0.05;
gap_threshold = 0.01;
supply_iter = zeros(length(d), 1);
iter_num = 0;
his_threshold = [];
his_step_sizes = [];

%% construct convex problem - dual decomposition
while (max(d - supply_iter') > gap_threshold) && (iter_num < 50)
    cvx_begin quiet
      variable pt1(length(d), 1);
      variable u_sep1(length(d)-1,1);

      minimize(sum(alpha(1)*pt1) + beta(1)*(pt1')*pt1 - Qt*pt1 + gamma(1)*sum(u_sep1));
      subject to
        u_sep1 >= pt1(1:end-1) - pt1(2:end);
        u_sep1 >= pt1(2:end) - pt1(1:end-1);
        pt1 <= Pmax(1);
        pt1 >= Pmin(1);
        u_sep1 <= R(1);
    cvx_end

    cvx_begin quiet
      variable pt2(length(d), 1);
      variable u_sep2(length(d)-1,1);

      minimize(sum(alpha(2)*pt2) + beta(2)*(pt2')*pt2 - Qt*pt2 + gamma(2)*sum(u_sep2));
      subject to
        u_sep2 >= pt2(1:end-1) - pt2(2:end);
        u_sep2 >= pt2(2:end) - pt2(1:end-1);
        pt2 <= Pmax(2);
        pt2 >= Pmin(2);
        u_sep2 <= R(2);
    cvx_end

    cvx_begin quiet
      variable pt3(length(d), 1);
      variable u_sep3(length(d)-1,1);

      minimize(sum(alpha(3)*pt3) + beta(3)*(pt3')*pt3 - Qt*pt3 + gamma(3)*sum(u_sep3));
      subject to
        u_sep3 >= pt3(1:end-1) - pt3(2:end);
        u_sep3 >= pt3(2:end) - pt3(1:end-1);
        pt3 <= Pmax(3);
        pt3 >= Pmin(3);
        u_sep3 <= R(3);
    cvx_end

    cvx_begin quiet
      variable pt4(length(d), 1);
      variable u_sep4(length(d)-1,1);

      minimize(sum(alpha(4)*pt4) + beta(4)*(pt4')*pt4 - Qt*pt4 + gamma(4)*sum(u_sep4));
      subject to
        u_sep4 >= pt4(1:end-1) - pt4(2:end);
        u_sep4 >= pt4(2:end) - pt4(1:end-1);
        pt4 <= Pmax(4);
        pt4 >= Pmin(4);
        u_sep4 <= R(4);
    cvx_end

    % check gap
    iter_num = iter_num + 1;
    supply_iter = pt1 + pt2 + pt3 + pt4;
    Qt = Qt + step_size*(d-supply_iter');
    max_threshold = max(d - supply_iter');
    if max_threshold <= 1
        step_size = 0.01;
        if max_threshold <= 0.5
            step_size = 0.005;
        end
    end
    his_threshold = [his_threshold, max_threshold];
    his_step_sizes = [his_step_sizes, step_size];
    fprintf('iteration: %d, maximum gap: %d\n', iter_num, max_threshold);
    fprintf('**********************************************************\n');
end

%% separate optimization as comparison - for validation
cvx_begin
  variable pt1_sep(length(d), 1);
  variable u1_sep(length(d)-1,1);

  minimize(sum(alpha(1)*pt1_sep) + beta(1)*(pt1_sep')*pt1_sep + gamma(1)*sum(u1_sep) - Qt*pt1_sep);
  subject to
    u1_sep >= pt1_sep(1:end-1) - pt1_sep(2:end);
    u1_sep >= pt1_sep(2:end) - pt1_sep(1:end-1);
    pt1_sep <= Pmax(1);
    pt1_sep >= Pmin(1);    
    u1_sep <= R(1);
cvx_end

%% plotting code; replace p and Q below with correct values
%p = ones(n,T);  % generator powers
Q = Qt;  % prices
p = [pt1'; pt2'; pt3'; pt4'];
subplot(3,1,1)
plot(t,d);
title('demand')
subplot(3,1,2)
plot(t,p);
title('generator powers')
subplot(3,1,3)
plot(t,Q);
title('power prices')