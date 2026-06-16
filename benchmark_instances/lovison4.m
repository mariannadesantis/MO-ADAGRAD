function [q,n,expensive,lb,ub,starting_points,f1,f2] = lovison4(y)

% from collection of test problems from 'Direct Multisearch for
% Multiobjective Optimization' by A. Custodio, J. Madiera, A. Vaz, L.
% Vicente, SIAM J. Optim., 21(3): 1109-1140, 2011.
% original paper:  A. Lovison in "A synthetic approach to multiobjective 
% optimization", arxiv Item: http://arxiv.org/abs/1002.0093.
% Example 4

q = 2;
n = 2;
expensive=1;

f1 = @(x) x(1)^2 + x(2)^2 + 4*( exp(-(x(1)+2)^2-x(2)^2) + exp(-(x(1)-2)^2-x(2)^2) );
f2 = @(x) (x(1)-6)^2 + (x(2)+0.5)^2;

lb = [0;-1];
ub = [6;1];

starting_points = zeros(10,n);
starting_points(:,1) = [4.1310    0.0971    1.1050    4.7880    5.3449    1.7134    4.7925    5.2571    1.1353    2.4022]';
starting_points(:,2) = [0.3978    0.3016   -0.7318    0.4300    0.8027    0.2935   -0.7054   -0.3587   -0.0336    0.5774]';

end