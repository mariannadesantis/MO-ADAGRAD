function [q,n,expensive,lb,ub,starting_points,f1,f2] = lovison3_unc(y)

% from collection of test problems from 'Direct Multisearch for
% Multiobjective Optimization' by A. Custodio, J. Madiera, A. Vaz, L.
% Vicente, SIAM J. Optim., 21(3): 1109-1140, 2011.
% original paper:  A. Lovison in "A synthetic approach to multiobjective 
% optimization", arxiv Item: http://arxiv.org/abs/1002.0093.
% Example 3

q = 2;
n = 2;
expensive=1;

f1 = @(x) x(1)^2 + x(2)^2;
f2 = @(x) (x(1)-6)^2 - (x(2)+0.3)^2;

lb = -Inf(n,1);
ub = Inf(n,1);

starting_points = zeros(10,n);
starting_points(:,1) = [10.4551   28.6764   36.3140    3.7379   10.0481  -36.5616  -35.4506   -2.9738   22.5676   24.3334]';
starting_points(:,2) = [-15.3237  -44.1752   -8.3337   -2.6164  -44.3159   12.8000  -12.1613  -16.9468   12.4817  -16.8038]';

end