function [f,g,h] = T2fun1(x)

f = sin(x(2));

if nargout >= 2 % gradient
    syms a b;
    f_help = sin(b);
    grad = gradient(f_help,[a,b]);
    g = double(subs(grad, {a,b}, {x(1),x(2)}));
    if nargout >= 3 % hessian
        h_help = jacobian(grad, [a,b]);
        h = double(subs(h_help, {a,b}, {x(1),x(2)}));
    end
end

end

