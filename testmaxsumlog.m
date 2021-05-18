%
%  test_maxsumlog: Test your maxsumlog algorithm
%     
%     
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

rng(1)
n=5;                       % Try different values 02 15 005 020, num of constraints
m=100;                       % Try different values 10 10 100 100, num of variables
A = randi([0, 20], n, m);
b = randi([1, 20], n, 1);
c = randi([0, 10], m, 1);

tic
[x,obj,y,it] = maxsumlog(A, b, c);
toc
sprintf('Number of iterations: %i',it)
sprintf('Number of non-zeros in x: %i',sum(x>1e-5))

% CVX solution for comparison
tic
cvx_begin quiet
  variable xcvx(m,1);
  dual variable y1;
  dual variable y2;
  maximize sum_log(1+xcvx.*c);
  subject to
    A*xcvx <= b : y1;
    xcvx>= 0 : y2;
cvx_end
toc

sprintf('Relative difference in x: %g',norm(x-xcvx)/norm(xcvx))
sprintf('Relative difference in objective: %g',...
        (obj-cvx_optval)/cvx_optval)
sprintf('Relative difference in y: %g',norm(y-[y1;y2])/norm([y1;y2]))

