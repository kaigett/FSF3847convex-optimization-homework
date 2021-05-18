function [x,obj,y,it] = maxsumlog(A, b, c, L0, MU, C_stop)

%
% [x,obj,y,it] = maxsumlog(A, b, c)
%	
%	A	- n x m matrix with non-negative entries
%	b	- n x 1 vector with positive entries
%	c	- m x 1 vector with non-negative entries
%	
% 	x	- Optimal solution
% 	obj	- Optimal objective value
% 	y	- Dual variables corresponding to Ax<=b and x>=0
% 	it	- Number of iterations
%	
%
%  maxsumlog: Solves
%     minimize 	sum_i log(1+c_i x_i)
%     s.t.	Ax <= b
%     		x >= 0
%     
%% define initial primal feasible points
n = size(A, 1);
m = size(A, 2);

% hyperparameters
if nargin == 3
    L0 = 1;   % initial lambda value, dual
    MU = 5;     % MU > 0, for t update, control iteration speed
    C_stop = 1e-5;  % stopping criteria
end
X0 = 0;  % initial x value, primal
epi = 0;    % episode number
STEP_CHANGE = 0.99; % control the step size of searching feasible step

% inital primal and dual feasible points
x = X0*ones(length(c), 1);  % initial x, primal
lambda1 = L0*ones(length(b), 1);
lambda2 = L0*ones(length(c), 1);
lambda = [lambda1; lambda2];    % initial lambda, dual

% calculate surrogate duality gap
h_x1 = b - A*x;
h_x2 = x;
h_x = [h_x1; h_x2];
eta = (h_x')*lambda;

%% primal-dual update
while (eta > C_stop)% && (epi <= 100)
    t = MU*length(lambda)/eta;  % 
    epi = epi + 1;
    
    % Newton step: Matrix_R*[delta_x; delta_lambda] = R
    % https://www.stat.cmu.edu/~ryantibs/convexopt-F15/lectures/16-primal-dual.pdf
    
    % construct Matrix_R = [A, B; C, D] for Newton step calculation
    % gradient and Hessian of f(x)
    diff_x = c./(c.*x+1);   % n*1
    diff2_x = -diag((c.^2/(c'*x+1).^2));    % n*n
    % Hessian of h(x), from observation, they are all zeros!
    diff2_hx = zeros(m, m);
    A_matrix = diff2_x + diff2_hx;     % n*n
    % gradient of h(x)
    diff_hx1 = -A;
    diff_hx2 = eye(m);
    diff_hx = [diff_hx1; diff_hx2];    % l*n
    B_matrix = diff_hx;     % l*n
    C_matrix = diag(lambda)*diff_hx;   % l*n
    D_matrix = diag(h_x);   % l*l
    Matrix_R = [A_matrix, B_matrix'; C_matrix, D_matrix];

    % construct R = [R_1; R_2] for Newton step calculation
    R_1 = diff_x + (diff_hx')*lambda;
    R_2 = diag(lambda)*h_x - 1/t;
    R = [R_1; R_2];
    
    % perform linsolve to get update direction
    delta = linsolve(Matrix_R, -R);
    
    % acquire delta_x and delta_lambda
    delta_x = delta(1: m);
    delta_lambda1 = delta(m+1: m+n);
    delta_lambda2 = delta(m+n+1: end);
    delta_lambda = delta(m+1: end);
    
    % search step size
    step_size = 1;  % initial max step is set to 1
    % iterate steps until all h(x) > 0 and lambda > 0 (from perturbed KKT reqs)
    while max(((b - A*(x+step_size*delta_x)) < 0)) || max((x+step_size*delta_x < 0)) || max((lambda+step_size*delta_lambda < 0))
        step_size = STEP_CHANGE*step_size;
    end
    
    % update primal and dual values
    x = x + step_size*delta_x;
    lambda1 = lambda1 + step_size*delta_lambda1;
    lambda2 = lambda2 + step_size*delta_lambda2;
    lambda = [lambda1; lambda2];
    
    %% check the gap
    h_x1 = b - A*x;
    h_x2 = x;
    h_x = [h_x1; h_x2];
    eta = (h_x')*lambda;
    fprintf('iteration: %d, error: %d\n', epi, eta);
    % print more details if you like
%     px = sprintf('%1.2f ', x);
%     fprintf('x = %s\n', px);
%     pdeltax = sprintf('%1.2f ', delta_x);
%     fprintf('delta_x = %s\n', pdeltax);
%     plambda = sprintf('%1.2f ', lambda);
%     fprintf('lambda1 = %s\n', plambda);
%     pdeltalambda = sprintf('%1.2f ', delta_lambda);
%     fprintf('delta_lambda1 = %s\n', pdeltalambda);
%     fprintf('*****************************************\n');
end
% check objective value
obj = sum(log(1 + x.*c));
y = lambda;
it = epi;
end
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% The above is a suggested interface. Feel free to add more parameters. 

% Write your Matlab routine below. 
% Print out one line of progress information per iteration. 

