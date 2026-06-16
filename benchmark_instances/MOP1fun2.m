function [f,g,h] = MOP1fun2(x)

% function value
f = (x(1)-2).^2;

if nargout >= 2 % gradient
    syms a; 
    f_help = (a-2).^2;
    grad = gradient(f_help,a);
    g = double(subs(grad, {a}, {x(1)}));
    if nargout >= 3
        h_help = jacobian(grad,a);
        h = double(subs(h_help, {a}, {x(1)}));
    end
end

end
