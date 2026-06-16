function [q,n,expensive,lb,ub,starting_points,f1,f2] = T2(y)

% = T2
% self-chosen

q = 2;
n = 2;
expensive = 2;

f1 = @(x) sin(x(2));
f2 = @(x) 1 - exp( -( x(1) - sqrt(2)^(-1) ).^2 -( x(2) - sqrt(2)^(-1) ).^2 );

lb = -Inf(n,1);
ub = Inf(n,1);


starting_points = zeros(10,n);
starting_points = [20.6046  -22.3077  -40.2868  -0.3985   45.0222   -6.1256   -0.0112  -31.3127   0.5105   0.9018;
  -46.8167  -45.3829   32.3458  0.9981  -46.5554  -11.8442   -0.5104   -1.0236   1.0012   1.5112]';


% ideal_point = [-1 0]; % ideal point

end