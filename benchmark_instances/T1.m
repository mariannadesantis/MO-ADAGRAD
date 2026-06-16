function [q,n,expensive,lb,ub,starting_points,f1,f2] = T1(y)

% = T1
% self-chosen


q = 2;
n = 2;
expensive = 1;

f1 = @(x) 0.5*x(1).^2 + x(2).^2 - 10*x(1) - 100;
f2 = @(x) x(1).^2 + 0.5*x(2).^2 - 10*x(2) - 100;

lb = -Inf(2,1);
ub = Inf(2,1);

starting_points = zeros(10,n);
starting_points = [-34.2387   45.7167   30.0280   -7.8239   29.2207   15.5741   34.9129   17.8735   24.3132   15.5478;
   47.0593   -1.4624  -35.8114   41.5736   45.9492  -46.4288   43.3993   25.7740  -10.7773  -32.8813]';

end