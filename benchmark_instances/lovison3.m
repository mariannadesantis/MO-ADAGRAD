function [q,n,expensive,lb,ub,starting_points,f1,f2] = lovison3(y)

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

lb = [0;-4];
ub = [6;4];

starting_points = zeros(10,n);
starting_points(:,1) = [1.3158    5.8931    5.8811    1.4557    4.5694    0.2416    3.4611    4.2050    3.1811    2.9385]';
starting_points(:,2) = [-2.0845   -0.9588   -2.2810    0.5870    3.8155   -2.8964   -3.1304    0.7960   -0.2692    3.2582]';

end