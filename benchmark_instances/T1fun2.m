function [f,g,h] = T1fun2(x)

f = x(1).^2 + 0.5*x(2).^2 - 10*x(2) - 100;

if nargout >= 2 % gradient
    syms a b;
    f_help = a.^2 + 0.5*b.^2 - 10*b - 100;
    grad = gradient(f_help,[a,b]);
    g = double(subs(grad, {a,b}, {x(1),x(2)}));
    if nargout >= 3 % hessian
        h_help = jacobian(grad, [a,b]);
        h = double(subs(h_help, {a,b}, {x(1),x(2)}));
    end
end

end

