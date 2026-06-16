function [q,n,expensive,lb,ub,starting_points,f1,f2] = MOP1(y)

% source: 
% Simon Huband, Phil Hingston, Luigi Barone, and Lyndon While. “A Review of
% Multiobjective Test Problems and a Scalable Test Problem Toolkit”. In: 
% IEEE Trans. Evolutionary Computation 10.5 (2006), pp. 477–506 

q = 2;
n = 1;
expensive = 1;

f1 = @(x) x(1).^2;
f2 = @(x) (x(1)-2).^2;

lb = -Inf;
ub = Inf;

starting_points = zeros(10,n);
starting_points = [ -8.2733  -45.0346   40.2716   44.4787   -0.9136   -1.0747  -16.2281   40.0054  -13.0753  -38.8797]';

% ideal_point = [0 0]; % ideal point

end