function [x, norm_gs, f12, i, eff_meas, flag, gs_flag, total_time] = ...
    MO_Methods(fun1, fun2, x0, param)
% Implementation of the MO-Descent [2] and MO-Adagrad [1] algorithms for
% bi-objective instances.

% References:
% [1] M. De Santis, G. Eichfelder. M. Porcelli,
% "Objective-Function Free Multi-Objective Optimization: Rate of Convergence
% and Performance of an Adagrad-like algorithm",
% pp. 1-21, 2026 (arXiv:2602:05893).
% [2] J. Fliege, B. F. Svaiter,
% "Steepest descent methods for multicriteria optimization",
% Mathematical Methods of Operations Research, 51(3):479–494, 2000.


%****** Input ******
%   fun1    : fist objective function
%   fun2    : second objective function
%   x0      : starting point
%   param   : structure, containing the parameter of the model
%       .varsigma = constant value in the denominator of Adagrad weights
%       .maxit  = maximum number of iterations (suggested: 100000)
%       .tol    = tolerance on the criticality measure (suggested: 1.e-4)
%       .noise  = level of noise (e.g: 0, 0.05, 0.15, 0.25)
%       .maxtime = time limit in seconds
%
% ****** Output ******
%   x          : solution of the problem
%   norm_gs    : vector containing the criticality measure along iterations
%                ||gs||
%   f12        : a matrix containing the value of fun1 and fun2 in the first
%                and second column, resp., along the iterations
%   i          : employed number of iterations
%   eff_meas   : vector containing the employed number of iterations (i) and
%                the employed number of function evaluations (1 function
%                evaluation correspond to evaluate fun1 and fun2)
%   flag       : = 1 (converge test satisfied)
%                = 0 (too many function+gradient evaluations)
%                = -1 (time limit exceeded)
%   gs_flag    : exit flag of fmincon in the computation of the common
%                descent direction gs
%   total_time : elapsed time (tic/toc)
% ********************

n = length(x0);
m = 2;
x = x0;

printout =  false;

%Constant value in the denominator of the Adagrad weights:
varsigma = param.varsigma;

%Initialize variables:
norm_gs = zeros(param.maxit,1);
f12 = zeros(param.maxit,m);
sum_gs = 0; nfeval = 0;
start_time = tic; total_time = 0;
eff_meas = [NaN,NaN]; s_k = NaN;
gs_flag =[]; flag = [];

for i=1:param.maxit

    %Evaluate objective functions and gradients (no noise)
    [f1, g_k(:,1)]=fun1(x); %real gradient first objective
    [f2, g_k(:,2)]=fun2(x); %real gradient second objective

    if printout
        [gs_k_noisefree] = comp_dir(g_k);
        f2ols = f2;
    end

    % Add noise (param.tau - level of noise) to the objective functions
    % and to the gradients
    f1 = f1*(1+ param.tauf*randn(1));
    f2 = f2*(1+ param.tauf*randn(1));
    g_k(:,1) = g_k(:,1).*(ones(n,1) + param.tau*randn(n,1));
    g_k(:,2) = g_k(:,2).*(ones(n,1) + param.tau*randn(n,1));

    %Compute the gs_k common descent direction and its norm
    [gs_k, gs_flag] = comp_dir(g_k);
    norm_gs(i)=norm(gs_k);
    f12(i,1:m) = [f1 f2];

    if printout
        % Print some info at every iteration:
        fprintf(' iter = %d, max(f1,f2) = %e, f1=  %e, ng1 = %e, f2 =  %e, ng2 = %e, ||gs|| = %e, ||step|| = %e\n',...
            i,   max(f1,f2), f1, norm(g_k(:,1)), f2, norm(g_k(:,2)),  norm(gs_k), norm(s_k))
        %    fprintf(' iter = %d, max(f1,f2) = %e, f1=  %e, ng1 = %e, f2 =  %e, ng2 = %e  ||gs|| = %e, ||gs_true|| = %e\n',...
        % i,   max(f1,f2), f1, norm(g_k(:,1)), f2, norm(g_k(:,2)),  norm(gs_k), norm(gs_k_noisefree))
    end

    %Check stopping condition:
    if norm_gs(i)<param.tol
        % fprintf('tot iteration %d', i)
        eff_meas(1) = i;
        eff_meas(2) = nfeval;
        flag = 1;
        break
    end

    %Compute the weights for the stepsize
    sum_gs = sum_gs + norm(gs_k)^2;
    w_k=(varsigma+sum_gs).^(1/2);

    %Compute the stepsize:
    % MO-OFFO-like:
    if strcmp(param.solver,'MOFFO')
        s_k=-gs_k./w_k;

    else
        % MO Armijo line search-like:
        [alpha_k, nvf] = ls(-gs_k,x, fun1, fun2,f1,f2,g_k,param);

        %Update variable:
        s_k = -alpha_k*gs_k;

        %Update the number of function evaluations:
        nfeval = nfeval + nvf;
    end

    %Update variable:
    x=x+s_k;

    %Check: stop for failure (too many function+gradient evaluations)
    if (i+round(nfeval/n)) >= param.maxit
        eff_meas(1) = NaN;
        eff_meas(2) = NaN;
        flag = 0;
        break
    end

    %Check: stop for failure (time limit exceeded)
    total_time= toc(start_time);
    if total_time >= param.maxtime
        flag = -1;
        break
    end
end




end

%%% Computation of the common descent direction solving Omega(x) %%%
function [gs,exitflag] = comp_dir(g)

m = size(g,2);

obj = @(lambda) norm(g*lambda)^2;
lambda_0 = eye(m,1);

% Solve Omega(x) using fmincon:
options = optimoptions('fmincon','Display','off','Algorithm','sqp');
%hoptions = optimoptions('fmincon','Display','iter','Algorithm','sqp');
[lambda_s,fval,exitflag,output] = fmincon(obj,lambda_0,[],[],ones(m,1)',1,zeros(m,1),[],[], options);

if exitflag~= 1
    if(output.firstorderopt<10^-5)
        exitflag = 1;
        %disp(output.message)
    end
end

%Compute g_s^k:
gs = g*lambda_s;

end


%%% MO Armijo line search %%%
function [step,nfv] = ls(gks,x, fun1, fun2,f1,f2,g,param)

tauf = param.tauf;
t = 1;
beta = 0.1;

%Evaluate objective functions and gradients:
[f1t, g_kt(:,1)]=fun1(x+t*gks); % first objective
[f2t, g_kt(:,2)]=fun2(x+t*gks); % second objective

% Add noise to the objective functions:
f1t = f1t*(1 + tauf*randn(1));
f2t = f2t*(1 + tauf*randn(1));
nfv = 1;

c1 = f1 + beta*t*g(:,1)'*gks;
c2 = f2 + beta*t*g(:,2)'*gks;

% Armijo line search loop
while (((f1t>c1)||(f2t>c2))&&(t>10^-13))
    t = 0.5*t;
    [f1t, g_kt(:,1)]=fun1(x+t*gks);
    [f2t, g_kt(:,2)]=fun2(x+t*gks);
    % Add noise to the objective functions
    f1t = f1t*(1 + tauf*randn(1));
    f2t = f2t*(1 + tauf*randn(1));
    nfv = nfv + 1;

    c1 = f1 + beta*t*g(:,1)'*gks;
    c2 = f2 + beta*t*g(:,2)'*gks;
end

% stepsize:
step = t;

end
