% Main file to run the MO-Descent [2] and MO-Adagrad [1] on a bi-objective 
% multi-task geometric classification problem: Quadrants-Circle

% N points in the square [-2,2]^2 are labeled with respect to two different
% criteria: 
% Task1: to belong the main diagonal (Quadrants 1 and 3) and the 
%        anti-diagonal (Quadrants 2 and 4) (binary classification task);
% Task2: to be inside or outside the unit circle centered in the origin
%        (binary classification task).

% References:
% [1] M. De Santis, G. Eichfelder. M. Porcelli, 
% "Objective-Function Free Multi-Objective Optimization: Rate of Convergence 
% and Performance of an Adagrad-like algorithm", 
% pp. 1-21, 2026 (arXiv:2602:05893). 
% [2] J. Fliege, B. F. Svaiter,
% "Steepest descent methods for multicriteria optimization",
% Mathematical Methods of Operations Research, 51(3):479–494, 2000.
_
%
% Authors: M. De Santis, G. Eichfelder, M. Porcelli
% Date: February 2026

% clear; clc;
close all; clear all
% Choose the solver:
% LS (for MO-Descent Armijo line search) - MOFFO (MO-Adagrad)
param.solver = 'MOFFO';
param.solver = 'LS';

global tauf tau
tauf = 0; tau = 0; % no noise in the objectives/gradients

%% 1. Data configuration
rng(42);
N = 10000;
X_raw = -2 + 4 * rand(N, 2);

% --- LABELS ---
% Task 1: Quadrants (1, 2, 3, 4)
y_task1 = zeros(N, 1);
y_task1(X_raw(:,1)>=0 & X_raw(:,2)>=0) = 1;
y_task1(X_raw(:,1)<0  & X_raw(:,2)>=0) = 2;
y_task1(X_raw(:,1)<0  & X_raw(:,2)<0)  = 3;
y_task1(X_raw(:,1)>=0 & X_raw(:,2)<0)  = 4;

% Task 2: Circle (Inside = 1, Outside = 0)
y_task2 = double(sum(X_raw.^2, 2) <= 1);

global X_s y1_s y2_s N_train num_features
global X_test yt1_test yt2_test

% --- FEATURE ENGINEERING ---
% We retain the quadratic features to solve the circle problem
% Features: [1, x, y, x^2, y^2]
X = [ones(N, 1), X_raw, X_raw.^2];
num_features = size(X, 2);

%% 2. Data Splitting
N_train = 8000; N_val = 0; N_test = 2000;
p = randperm(N);

idx_T = p(1:N_train);
idx_V = p(N_train+1 : N_train+N_val);
idx_P = p(N_train+N_val+1 : end);

X_train = X(idx_T, :); yt1_train = y_task1(idx_T); yt2_train = y_task2(idx_T);
X_val   = X(idx_V, :); yt1_val   = y_task1(idx_V); yt2_val   = y_task2(idx_V);
X_test  = X(idx_P, :); yt1_test  = y_task1(idx_P); yt2_test  = y_task2(idx_P);

%% 3. Parameter initialization (Two weight vectors)
epochs = 500;
W1 = randn(num_features, 4) * 0.01; % Weights for Task 1
w2 = randn(num_features, 1) * 0.01; % Weights for Task 2
x = [W1(:); w2]; sum_gs = 0; nfeval=0;
loss_history = [];
zz = zeros(num_features, 1);
zz1 = zeros(num_features*4, 1);

%% 4. Training Loop 
% Prediction models: 
% Softmax Regression for Task1 and the Logistic Regression for Task2
% Loss function: categorical Cross-Entropy and the binary Cross-Entropy 
fprintf('Starting Training (Quadrants and Circle)...\n');
X_s = X_train;
y1_s = yt1_train;
y2_s = yt2_train;
start = tic;
for ep = 1:epochs

    [f1, g_k(:,1)]=fun1(x); 
    [f2, g_k(:,2)]=fun2(x); 
    f1 = f1*(1+ tauf*randn(1));
    f2 = f2*(1+ tauf*randn(1));

    g_k(:,1) = g_k(:,1).*(ones(5*num_features,1) + tau*randn(5*num_features,1));
    g_k(:,2) = g_k(:,2).*(ones(5*num_features,1) + tau*randn(5*num_features,1));

    [gs_k, gs_flag] = comp_dir(g_k);

    norm_gs(ep)=norm(gs_k);
    %f12(i,1:m) = [f1 f2];

    sum_gs = sum_gs + norm(gs_k)^2;
    w_k=(1e-2+sum_gs).^(1/2);


    %f1 = fun1(x); f2 = fun2(x);
    %Compute the step
    if strcmp(param.solver,'MOFFO')
        s_k=-gs_k./w_k;
        %Update variable
        %1/w_k
        x=x+s_k;
    else

        [alpha_k, nvf] = ls(-gs_k,x, @fun1, @fun2,f1,f2,g_k,param);
        %nvf
        %Update variable
        x=x-alpha_k*gs_k;
        %s_k = -alpha_k*gs_k;

        nfeval = nfeval + nvf;
    end


    %x=x+s_k;
    w1 = x(1:num_features*4);
    W1 = reshape(w1, num_features,4);
    w2 = x(num_features*4+1:end);
    if norm_gs(ep)<1e-3
        % % fprintf('tot iteration %d', i)
        %  eff_meas(1) = i;
        %  eff_meas(2) = nfeval;
        flag = 1;
        break
    end

    ep_loss = max(f1,f2);
    avg_loss = ep_loss / N_train;
    loss_history = [loss_history; avg_loss];

    %if mod(ep, 1) == 0
    [acc1,acc2]= compute_acc(W1,w2);
    % fprintf('Epoch %d/%d - max( f1, f2): %.4f ||gs|| %e nvf= %.0f acc1 =%.2f acc2 =%.2f\n', ...
    %    ep, epochs, avg_loss, norm_gs(ep), nfeval, acc1*100, acc2*100);
    minacc(ep,1:2) = [acc1,acc2];
    tempo(ep) = toc(start);

    %if min(acc1,acc2)*100 >= 99.5
    % % fprintf('tot iteration %d', i)
    %  eff_meas(1) = i;
    %  eff_meas(2) = nfeval;
    %flag = 1;

    %break

    %end

    %end


end
ttot = toc(start);

%% 5.Evaluation and Accuracy (Testing)

% --- Task 1 (Quadrants) ---
scores_test = X_test * W1;
[~, p1_test] = max(scores_test, [], 2); % Class index with maximum score

% --- Task 2 (Circle) ---
probs_test = 1 ./ (1 + exp(-(X_test * w2)));
p2_test = double(probs_test >= 0.5);

% Accuracy
acc1 = mean(p1_test == yt1_test);
acc2 = mean(p2_test == yt2_test);

fprintf('\n--- Final Results (Test Set) ---\n');
fprintf('Accuracy Task 1 (Quadranti): %.2f%%\n', acc1 * 100);
fprintf('Accuracy Task 2 (Circles):   %.2f%%\n', acc2 * 100);

%% 6. Plots
figure('Color', 'w', 'Position', [100, 100, 1200, 500]);

% Plot Task 1
subplot(1, 2, 1);
gscatter(X_test(:,2), X_test(:,3), p1_test, 'rgbk', '.', 5);
title('Task 1: Quadrants (Predicted)');
xlabel('x'); ylabel('y'); axis equal; grid on;
legend('Q1','Q2','Q3','Q4','Location','bestoutside');

% Plot Task 2
subplot(1, 2, 2);
gscatter(X_test(:,2), X_test(:,3), p2_test, 'br', '.', 5);
hold on;
theta = linspace(0, 2*pi, 100); plot(cos(theta), sin(theta), 'k-', 'LineWidth', 1.5);
title('Task 2: Circle (Predicted)');
xlabel('x'); ylabel('y'); axis equal; grid on;
legend('Outside','Inside','x^2+y^2 = 1','Location','bestoutside');

% compute the maximum accuracy reached along the iterations, considering
% the minumum value for the 2 tasks
[acc_max, ep_am] = max(min(minacc,[],2));
fprintf([param.solver,': noise=%.2f, min(a1,a2) = %.2f, iter: %.0f, #feval = %.0f, time %.1f\n  '],...
    tau, acc_max*100, ep_am, nfeval, tempo(ep_am))
minacc(ep_am,:)

%%%%%%%%%%%%%%%

%%% Computation of the common descent direction solving Omega(x) %%%
function [gs,exitflag] = comp_dir(g)

% g n x m g(:,j) is the gradient of f_j(x)
% 
%
m = size(g,2);

obj = @(lambda) norm(g*lambda)^2;
lambda_0 = eye(m,1);

options = optimoptions('fmincon','Display','off','Algorithm','sqp');
%hoptions = optimoptions('fmincon','Display','iter','Algorithm','sqp');
[lambda_s, fval,exitflag,output] = fmincon(obj,lambda_0,[],[],ones(m,1)',1,zeros(m,1),[],[], options);

if exitflag~= 1
    if(output.firstorderopt<10^-5)
        exitflag = 1;
        %disp(output.message)
    end
end
gs = g*lambda_s;

end


%%%%%%%%%%%%%%%
function [step,nfv] = ls(gks,x, fun1, fun2,f1,f2,g,param)

global  tauf
t = 1;
%beta = 0.1;
beta = 1e-4;
%Evaluate objective functions and gradients:
[f1t, g_kt(:,1)]=fun1(x+t*gks);
[f2t, g_kt(:,2)]=fun2(x+t*gks);
% Add noise to the objective functions
f1t = f1t*(1 + tauf*randn(1));
f2t = f2t*(1 + tauf*randn(1));
nfv = 1;

c1 = f1 + beta*t*g(:,1)'*gks;
c2 = f2 + beta*t*g(:,2)'*gks;

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
step = t;

end

function [loss1_new, g1] = fun1(x)

global X_s y1_s N_train num_features

zz = zeros(num_features, 1);

w1 = x(1:num_features*4);
W1 = reshape(w1, num_features,4);
loss1_new = 0;
grad_W1_new = zeros(num_features,4);
for i = 1:N_train
    xi = X_s(i, :)';    % [5x1]
    y1_i = y1_s(i);     % 0 o 1

    % --- TASK 1: Softmax ---
    scores1 = W1' * xi;
    scores1 = scores1 - max(scores1); % Stabilità numerica
    probs1 = exp(scores1) / sum(exp(scores1));

    % Gradiente Task 1
    y1_onehot = zeros(4,1);
    y1_onehot(y1_i) = 1;
    grad_W1 = xi * (probs1 - y1_onehot)';

    % Loss Task 1 (solo per monitoraggio)
    loss1 = -log(probs1(y1_i) + 1e-8);
    grad_W1_new = grad_W1_new + grad_W1;

    loss1_new =  loss1_new + loss1;

end

g1 = [grad_W1_new(:) ;zz];

end

function [loss2_new,g2] = fun2(x)

global X_s y2_s N_train num_features

w2 = x(num_features*4+1:end);
zz1 = zeros(num_features*4, 1);
loss2_new = 0; grad_w2_new =zeros(num_features,1);

for i = 1:N_train
    xi = X_s(i, :)';    % [5x1]
    y2_i = y2_s(i);     % 0 or 1


    % --- TASK 2: Sigmoid (Circle) ---
    z2 = w2' * xi;
    h2 = 1 / (1 + exp(-z2));

    % Gradient Task 2
    grad_w2 = xi * (h2 - y2_i);

    % Loss Task 2
    p2_safe = max(min(h2, 1-1e-8), 1e-8);
    loss2 = -(y2_i*log(p2_safe) + (1-y2_i)*log(1-p2_safe));


    grad_w2_new = grad_w2_new + grad_w2;
    loss2_new =  loss2_new + loss2;

end

g2 = [zz1;grad_w2_new];
end


function [acc1,acc2]= compute_acc(W1,w2)

global X_test yt1_test yt2_test

% --- Task 1 (Quadrants) ---
scores_test = X_test * W1;
[~, p1_test] = max(scores_test, [], 2); 

% --- Task 2 (Circle) ---
probs_test = 1 ./ (1 + exp(-(X_test * w2)));
p2_test = double(probs_test >= 0.5);

% Accuracy
acc1 = mean(p1_test == yt1_test);
acc2 = mean(p2_test == yt2_test);

end