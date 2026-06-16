function [f,g,h] = T1fun1(x)

f = 0.5*x(1).^2 + x(2).^2 - 10*x(1) - 100;

if nargout >= 2 % gradient
    syms a b;
    f_help = 0.5*a.^2 + b.^2 - 10*a - 100;
    grad = gradient(f_help,[a,b]);
    g = double(subs(grad, {a,b}, {x(1),x(2)}));
    if nargout >= 3
        h_help = jacobian(grad, [a,b]);
        h = double(subs(h_help, {a,b}, {x(1),x(2)}));
    end
end

end

