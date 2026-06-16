function [f,g,h] = lovison4fun1(x)

% from collection of test problems from 'Direct Multisearch for
% Multiobjective Optimization' by A. Custodio, J. Madiera, A. Vaz, L.
% Vicente, SIAM J. Optim., 21(3): 1109-1140, 2011.
% original paper:  A. Lovison in "A synthetic approach to multiobjective 
% optimization", arxiv Item: http://arxiv.org/abs/1002.0093.
% Example 4

f = x(1)^2 + x(2)^2 + 4*( exp(-(x(1)+2)^2-x(2)^2) + exp(-(x(1)-2)^2-x(2)^2) );

if nargout >= 2 % gradient
    syms a b;
    f_help = a^2 + b^2 + 4*( exp(-(a+2)^2-b^2) + exp(-(a-2)^2-b^2) );
    grad = gradient(f_help,[a,b]);
    g = double(subs(grad, {a,b}, {x(1),x(2)}));
    if nargout >= 3
        h_help = jacobian(grad, [a,b]);
        h = double(subs(h_help, {a,b}, {x(1),x(2)}));
    end
end

end