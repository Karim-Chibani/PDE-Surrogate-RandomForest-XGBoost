% Script Complet : Analyse de convergence et d'erreur numérique
clear; clc; close all;

% Configuration du test
N_values = [20, 40, 80, 160];
tol = 1e-8; 
max_iter = 50;

% 1. Calcul de la solution de référence (Maillage très fin)
N_ref = 640; 
fprintf('Calcul de la solution de référence (N = %d)... \n', N_ref);

x_ref = linspace(0, 1, N_ref + 1)'; 
K_ref = stiffness_matrix(x_ref);
F_ref = rhs_vector(x_ref);
U_ref = newton(K_ref, F_ref, x_ref, tol, max_iter);

% Initialisation des vecteurs d'erreur
err_L2 = zeros(size(N_values));
err_H1 = zeros(size(N_values));

% 2. Boucle sur les différents maillages
for k = 1:length(N_values)
    N = N_values(k);
    fprintf('Simulation en cours pour N = %d... \n', N);
    
    x = linspace(0, 1, N + 1)'; 
    K = stiffness_matrix(x);
    F = rhs_vector(x);
    U = newton(K, F, x, tol, max_iter);
    
    % Évaluation de la solution courante sur le maillage de référence
    U_interp = reconstruction(x_ref, x, U);
    
    % Calcul de l'erreur L2
    diff_u = U_interp - U_ref;
    err_L2(k) = sqrt(trapz(x_ref, diff_u.^2));
    
    % Calcul de l'erreur H1 (semi-norme)
    h_ref = x_ref(2) - x_ref(1);
    grad_diff = diff(diff_u) / h_ref; 
    err_H1(k) = sqrt(trapz(x_ref(1:end-1), grad_diff.^2));
end

% 3. Calcul des ordres de convergence (EOC)
eoc_L2 = log(err_L2(1:end-1)./err_L2(2:end)) / log(2);
eoc_H1 = log(err_H1(1:end-1)./err_H1(2:end)) / log(2);

% 4. Affichage du tableau des résultats
fprintf('\n=========================================================\n');
fprintf('  N   |   Erreur L2   |  EOC L2  |   Erreur H1   |  EOC H1  \n');
fprintf('---------------------------------------------------------\n');
for k = 1:length(N_values)
    if k == 1
        fprintf('%3d   |  %10.4e  |   --     |  %10.4e  |   --    \n', N_values(k), err_L2(k), err_H1(k));
    else
        fprintf('%3d   |  %10.4e  |  %5.2f   |  %10.4e  |  %5.2f  \n', N_values(k), err_L2(k), eoc_L2(k-1), err_H1(k), eoc_H1(k-1));
    end
end
fprintf('=========================================================\n');

% 5. Tracé des courbes de convergence
% 5. Tracé des courbes de convergence
figure;
loglog(N_values, err_L2, '-o', 'LineWidth', 2, 'MarkerSize', 8); hold on; % Un seul marqueur 'o'
loglog(N_values, err_H1, '-s', 'LineWidth', 2, 'MarkerSize', 8);        % Un seul marqueur 's'

% Tracé des pentes théoriques pour comparaison
loglog(N_values, 0.1*(N_values.^-2), '--k', 'LineWidth', 1.5);
loglog(N_values, 0.5*(N_values.^-1), ':.k', 'LineWidth', 1.5);

ylabel('Erreur numérique');
title('Analyse de Convergence Éléments Finis (P1)');
legend('Erreur L^2', 'Erreur H^1 (semi-norme)', 'Pente théorique O(h^2)', 'Pente théorique O(h)', 'Location', 'SouthWest');


% =========================================================================
% FONCTIONS LOCALES (Intégrées directement pour éviter les conflits)
% =========================================================================

function K = stiffness_matrix(x)
    N = length(x) - 1;
    K = zeros(N+1, N+1);
    for i = 1:N
        h = x(i+1) - x(i);
        K_elem = [1/h, -1/h; -1/h, 1/h];
        K(i:i+1, i:i+1) = K(i:i+1, i:i+1) + K_elem;
    end
    K(1, :) = 0; K(1, 1) = 1;
    K(end, :) = 0; K(end, end) = 1;
end

function F = rhs_vector(x)
    N = length(x) - 1;
    F = zeros(N+1, 1);
    f = @(v) sin(pi * v);
    for i = 1:N
        h = x(i+1) - x(i);
        F(i)   = F(i)   + (h / 2) * f(x(i));
        F(i+1) = F(i+1) + (h / 2) * f(x(i+1));
    end
    F(1) = 0; F(end) = 0;
end

function R = residual(U, K, F, x)
    N = length(x) - 1;
    R = K * U - F;
    N_vector = zeros(N+1, 1);
    g_pts = [ -1/sqrt(3), 1/sqrt(3) ]; g_w = [ 1, 1 ];
    for i = 1:N
        h = x(i+1) - x(i);
        for q = 1:2
            xq = 0.5 * h * g_pts(q) + 0.5 * (x(i+1) + x(i));
            wq = 0.5 * h * g_w(q);
            u_q = U(i) * ((x(i+1) - xq)/h) + U(i+1) * ((xq - x(i))/h);
            N_vector(i)   = N_vector(i)   + wq * (u_q^3) * ((x(i+1) - xq)/h);
            N_vector(i+1) = N_vector(i+1) + wq * (u_q^3) * ((xq - x(i))/h);
        end
    end
    N_vector(1) = 0; N_vector(end) = 0;
    R = R + N_vector;
    R(1) = U(1); R(end) = U(end);
end

function J = jacobian(U, K, x)
    N = length(x) - 1;
    J = K;
    J_nl = zeros(N+1, N+1);
    g_pts = [ -1/sqrt(3), 1/sqrt(3) ]; g_w = [ 1, 1 ];
    for i = 1:N
        h = x(i+1) - x(i);
        for q = 1:2
            xq = 0.5 * h * g_pts(q) + 0.5 * (x(i+1) + x(i));
            wq = 0.5 * h * g_w(q);
            u_q = U(i) * ((x(i+1) - xq)/h) + U(i+1) * ((xq - x(i))/h);
            der_nl = 3 * (u_q^2);
            phi_i = (x(i+1) - xq)/h; phi_j = (xq - x(i))/h;
            J_nl(i, i)     = J_nl(i, i)     + wq * der_nl * phi_i * phi_i;
            J_nl(i, i+1)   = J_nl(i, i+1)   + wq * der_nl * phi_i * phi_j;
            J_nl(i+1, i)   = J_nl(i+1, i)   + wq * der_nl * phi_j * phi_i;
            J_nl(i+1, i+1) = J_nl(i+1, i+1) + wq * der_nl * phi_j * phi_j;
        end
    end
    J = J + J_nl;
    J(1, :) = 0; J(1, 1) = 1;
    J(end, :) = 0; J(end, end) = 1;
end

function [U, iter] = newton(K, F, x, tol, max_iter)
    U = zeros(length(x), 1);
    for iter = 1:max_iter
        R = residual(U, K, F, x);
        if norm(R, 2) < tol
            return;
        end
        J = jacobian(U, K, x);
        dU = - J \ R;
        U = U + dU;
    end
end

function [val, der] = hat_function(x, x_nodes, i)
    val = 0; der = 0;
    if i > 1 && x >= x_nodes(i-1) && x <= x_nodes(i)
        h = x_nodes(i) - x_nodes(i-1); val = (x - x_nodes(i-1)) / h; der = 1 / h;
    elseif i < length(x_nodes) && x >= x_nodes(i) && x <= x_nodes(i+1)
        h = x_nodes(i+1) - x_nodes(i); val = (x_nodes(i+1) - x) / h; der = -1 / h;
    end
end

function u_eval = reconstruction(x_eval, x_nodes, U)
    u_eval = zeros(size(x_eval));
    for j = 1:length(x_eval)
        val_total = 0;
        for i = 1:length(x_nodes)
            [val, ~] = hat_function(x_eval(j), x_nodes, i);
            val_total = val_total + U(i) * val;
        end
        u_eval(j) = val_total;
    end
end