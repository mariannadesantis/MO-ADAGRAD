% Main file to run the MO-Descent [2] and MO-Adagrad [1] on unconstrained 
% bemchmark bi-objectives instances [3] (references below).

% References:
% [1] M. De Santis, G. Eichfelder. M. Porcelli, 
% "Objective-Function Free Multi-Objective Optimization: Rate of Convergence 
% and Performance of an Adagrad-like algorithm", 
% pp. 1-21, 2026 (arXiv:2602:05893). 
% [2] J. Fliege, B. F. Svaiter,
% "Steepest descent methods for multicriteria optimization",
% Mathematical Methods of Operations Research, 51(3):479–494, 2000.
% [3] J. Thomann, G. Eichfelder,
% "A trust-region algorithm for heterogeneous multiobjective optimization",
% SIAM Journal on Optimization, 29(2):1017–1047, 2019.

%
% Authors: M. De Santis, G. Eichfelder, M. Porcelli
% Date: February 2026
close all
clear all

% Include the folder with instances:
addpath(genpath('./benchmark_instances'))

ff = zeros(10,2);
eff = zeros(10,1);

%Files to store results:
fileID = fopen('results_MObench.txt','a+');
fileID2 = fopen('res_num.txt','a+');
%Format for the results: solver, starting point, noise, g+f eval, f1, f2:
formatspec = "%d %d %.2f %.2f %.2f %.2f \r\n";

% List of instances:
list = {'lovison3_unc',
    'lovison4_unc',
    'MOP1',
    'T1',
    'T2'
    };

% Level of noise considered
valnoise = [0.0; 0.05; 0.15; 0.25];


for j = 1: size(list,1) % instance
    for noise = 1:4 % level of noise
        for solver = 1:2 % LS (for MO-Descent Armijo line search) - MOFFO (MO-Adagrad)
            for i = 1:10 % starting point
                starting_point = i; % (choose among pre-defined starting point)
                % If 'user', then define the surrogate ideal point with finite components:
                p = [Inf Inf];

                which_problem = list{j,1};
                help = str2func(which_problem);
                name_fun1 = [which_problem, 'fun1'];
                name_fun2 = [which_problem, 'fun2'];

                [q,n,expensive,lb,ub,starting_points,f1,f2] = help(0);
                fun1 = str2func(name_fun1);
                fun2 = str2func(name_fun2);

                number = size(starting_points,1);
                xstart = starting_points(starting_point,:);


                switch solver
                    case 1
                        param.solver = ['LS'];
                    case 2
                        param.solver = ['MOFFO'];
                end

                % Set parameters:
                param.tol = 1e-3;
                param.maxit = 1e5;
                param.maxtime = 7200;
                param.varsigma = 1e-12; %starting stepsize for MO-Adagrad
                param.tau = valnoise(noise); % level of noise (gradients)
                param.tauf = valnoise(noise); % level of noise (objective functions)

                %Call solver:
                [x,norm_g,f12,iters,eff_meas,flag, gs_flag, total_time] = ...
                    MO_Methods(fun1, fun2, xstart, param);

                ff(i,1:2 ) = [f12(iters,1), f12(iters,2)];
                % Normalize function evaluations for LS:
                eff(i) = eff_meas(1)+eff_meas(2)/2;

                % Store results:
                fprintf([which_problem,'_x0_', num2str(starting_point), '  ', param.solver, '_', num2str(param.tau),' weighted f/g eval = %4.1f,  ||g^s|| = %e, objs: f1 = %f, f2 = %f\n'],...
                    eff_meas(1)+eff_meas(2)/2, norm_g(iters), f12(iters,1), f12(iters,2))

                fprintf(fileID2,formatspec,solver,i,valnoise(noise), eff_meas(1)+eff_meas(2)/2,f12(iters,1), f12(iters,2));

                fprintf(fileID,[which_problem,'_x0_', num2str(starting_point), '  ', param.solver, '_', num2str(param.tau),' weighted f/g eval = %4.1f,  ||g^s|| = %e, objs: f1 = %f, f2 = %f\n'],...
                    eff_meas(1)+eff_meas(2)/2, norm_g(iters), f12(iters,1), f12(iters,2));



                if 0 % Prot results in the criterion space
                    plot_result_bi_MAD(which_problem, fun1, fun2, lb, ub, f12, iters);
                    figure
                    semilogy(norm_g(1:iters))
                    hold on
                    %semilogy(1./(2:length(norm_g)+1).^2)
                    legend('||g^s_k||')
                    xlabel('iterations')

                    figure
                    axis equal
                    hold on
                    plot(f12(1:iters,1),f12(1:iters,2),'*')
                    xlabel('f_1')
                    ylabel('f_2')

                    figure
                    %axis equal
                    hold on
                    plot(1:iters,max(f12(1:iters,1),f12(1:iters,2)),'or')
                    xlabel('iterations')
                    ylabel('max(f_1,f_2)')
                end

            end
        end
    end
end

fclose(fileID);
fclose(fileID2);
