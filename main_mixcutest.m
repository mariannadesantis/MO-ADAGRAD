% Main file to run the MO-Descent [2] and MO-Adagrad [1] on bi-objective 
% instances generated mixing CUTEst problems [3] (references below).

% References:
% [1] M. De Santis, G. Eichfelder. M. Porcelli, 
% "Objective-Function Free Multi-Objective Optimization: Rate of Convergence 
% and Performance of an Adagrad-like algorithm", 
% pp. 1-21, 2026 (arXiv:2602:05893). 
% [2] J. Fliege, B. F. Svaiter,
% "Steepest descent methods for multicriteria optimization",
% Mathematical Methods of Operations Research, 51(3):479–494, 2000.
% [3] S. Gratton, Ph. L. Toint,
% "S2MPJ and CUTEst optimization problems for Matlab, Python and Julia",
% Optimization Methods and Software, 40(4):871-903, 2025.

% The MATLAB function s2mpjlib.m from [3] is needed.
% See also https://github.com/GrattonToint/S2MPJ  

%
% Authors: M. De Santis, G. Eichfelder, M. Porcelli
% Date: February 2026

close all
clear all

% Include the folder with instances:
addpath ./mix_CUTEst_instances/

% File to store results:
fileID = fopen('results_mix_cutest.txt','a+');

% List of instances considered:
list = {'BROWNDEN','ALLINITU' %N= 4
    'BROWNAL','ARWHEAD', %N=10
    'BROWNAL','VARDIM', %N=10
    'ARWHEAD','VARDIM',%N=10
    'ZANGWIL2', 'ROSENBR',  %N= 2
    'ZANGWIL2', 'CUBE',%N= 2
    'ZANGWIL2', 'WAYSEA1', %N= 2  
    'ROSENBR','WAYSEA1',%N= 2
    'ROSENBR','CUBE',%N= 2
    'WAYSEA1','CUBE'%N= 2
    };

% Level of noise considered
valnoise = [0.0; 0.05; 0.15; 0.25];


for noise = 1:4 % Level of noise
    for solver = 1:2 % LS (for MO-Descent Armijo line search) - MOFFO (MO-Adagrad)
        for i = 1: size(list,1) % instance
            for rr = 1:1 % random seed
                rng(rr)

                name = str2func(list{i,1});

                % name = str2func(['BROWNDEN']);
                %% set up the problem
                [ pb, pbm ] = name( 'setup');

                name2 = str2func(list{i,2});
                [ pb2, pbm2 ] = name2( 'setup');

                %Initialization
                name_fun1 = pb.name;
                name_fun2 = pb2.name;
                which_problem = [name_fun1,' vs ',name_fun2];

                fun1 = @(x)name( 'fx',x);
                fun2 = @(x)name2( 'fx',x); % to mix cutest problems

                % Starting point:
                for st = 1:3 %(starting point of pb, pb2 or their middle point)
                    if st == 1
                        xstart = pb.x0;
                    else
                        if st == 2
                            xstart = pb2.x0;
                        else
                            if st == 3
                                xstart = (pb.x0+pb2.x0)/2;
                            end
                        end
                    end

                    switch solver
                        case 1
                            param.solver = ['LS'];
                        case 2
                            param.solver = ['MOFFO'];
                    end

                    param.tol = 1e-3;
                    param.maxit = 1e5;
                    param.maxtime = 7200;
                    param.varsigma = 1e-2; %starting stepsize
                    param.tau = valnoise(noise); %  level of noise (gradients)
                    param.tauf = valnoise(noise); %  level of noise (objective functions)

                    fprintf('Solving %12s & %12s with %s ... \n', ...
                        pb.name, pb2.name, param.solver)

                    % Call solver:
                    [x,norm_g,f12,iter,eff_meas,exitflag, gs_flag, total_time]= ...
                        MultiAdagrad(fun1, fun2, xstart, param);

                    % Save results
                    fprintf(fileID,'%5s - %8s/%8s n = %3d, sp = %3d, noise = %.2e, ||gs|| = %e, geval = %4d, feval = %4d, flag = %2d, gs_flag = %2d, time = %.2d, objs: f1 = %f, f2 = %f\n', ...
                        param.solver, pb.name, pb2.name, pb.n, st, param.tau, norm_g(iter), eff_meas(1), eff_meas(2), exitflag, gs_flag,total_time,f12(iter,1), f12(iter,2));

                    fprintf('%5s - %8s/%8s, n = %3d, , sp = %3d, noise = %.2e, ||gs|| = %e, geval = %4d feval = %4d flag = %2d gs_flag = %2d, time = %.2d objs: f1 = %f, f2 = %f\n', ...
                        param.solver, pb.name, pb2.name, pb.n, st, param.tau, norm_g(iter), eff_meas(1), eff_meas(2), exitflag, gs_flag,total_time,f12(iter,1), f12(iter,2));
                end

            end
        end

    end
end

fclose(fileID);

