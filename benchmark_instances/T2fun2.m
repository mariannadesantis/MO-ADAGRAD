function [f,g,h] = T2fun2(x)

f = 1 - exp( -( x(1) - sqrt(2)^(-1) ).^2 -( x(2) - sqrt(2)^(-1) ).^2 );

if nargout >= 2 % gradient
    syms a b;
    f_help = 1 - exp( -( a - sqrt(2)^(-1) ).^2 -( b - sqrt(2)^(-1) ).^2 );
    grad = gradient(f_help,[a,b]);
    g = double(subs(grad, {a,b}, {x(1),x(2)}));
    if nargout >= 3 % hessian        
        h_help = jacobian(grad, [a,b]);
        h = double(subs(h_help, {a,b}, {x(1),x(2)}));
    end
end

end