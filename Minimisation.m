%Script : Étude Énergétique - Minimisation de J(u)
clear; clc; close all;

% Configuration du test
N_values = [10, 20, 40, 80, 160, 320];
tol = 1e-8; 
max_iter = 50;

% Vecteur pour stocker l'énergie J(u_h)
J_values = zeros(size(N_values));

% Boucle sur les maillages
for k = 1:length(N_values)
    N = N_values(k);
    
    % 1. Maillage et résolution
    x = linspace(0, 1, N + 1)'; 
    K = stiffness_matrix(x);
    F = rhs_vector(x);
    U = newton(K, F, x, tol, max_iter);
    
    % 2. Calcul de l'énergie J(U)
    % Terme 1 : 0.5 * \int |u'|^2 dx  =>  0.5 * U' * K * U
    E_lin = 0.5 * (U' * K * U);
    
    % Terme 2 : 0.25 * \int u^4 dx  (Intégration par méthode des trapèzes)
    E_nonlin = 0;
    for i = 1:N
        h = x(i+1) - x(i);
        % Approximation de l'intégrale de u^4 sur chaque élément [x_i, x_{i+1}]
        E_nonlin = E_nonlin + (h / 2) * (U(i)^4 + U(i+1)^4);
    end
    E_nonlin = 0.25 * E_nonlin;
    
    % Terme 3 : \int f * u dx  =>  U' * F
    E_source = U' * F;
    
    % Énergie totale J(u_h)
    J_values(k) = E_lin + E_nonlin - E_source;
    
    fprintf('Pour N = %3d | Énergie J(u_h) = %12.8f\n', N, J_values(k));
end

% 3. Affichage du tableau des énergies
fprintf('\n===================================\n');
fprintf('  N   |    Énergie J(u_h)  \n');
fprintf('-----------------------------------\n');
for k = 1:length(N_values)
    fprintf('%3d   |    %12.8f  \n', N_values(k), J_values(k));
end
fprintf('===================================\n');

% 4. Tracé de l'évolution de l'énergie
figure;
plot(N_values, J_values, '-ro', 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', 'r');
grid on;
xlabel('Nombre d''éléments (N)');
ylabel('Énergie J(u_h)');
title('Convergence de l''énergie J(u_h) en fonction de N');

% =========================================================================
% FONCTIONS LOCALES
% =========================================================================
function K = stiffness_matrix(x)
    N = length(x) - 1; K = zeros(N+1, N+1);
    for i = 1:N
        h = x(i+1) - x(i); K_elem = [1/h, -1/h; -1/h, 1/h];
        K(i:i+1, i:i+1) = K(i:i+1, i:i+1) + K_elem;
    end
    K(1, :) = 0; K(1, 1) = 1; K(end, :) = 0; K(end, end) = 1;
end

function F = rhs_vector(x)
    N = length(x) - 1; F = zeros(N+1, 1); f = @(v) sin(pi * v);
    for i = 1:N
        h = x(i+1) - x(i);
        F(i)   = F(i)   + (h / 2) * f(x(i));
        F(i+1) = F(i+1) + (h / 2) * f(x(i+1));
    end
    F(1) = 0; F(end) = 0;
end

function [U, iter] = newton(K, F, x, tol, max_iter)
    U = zeros(length(x), 1);
    for iter = 1:max_iter
        R = K*U - F; N_vec = zeros(length(x), 1);
        for i = 1:length(x)-1
            h = x(i+1)-x(i); u_avg = 0.5*(U(i)+U(i+1)); % approximation rapide pour Newton
            N_vec(i) = N_vec(i) + (h/2)*(u_avg^3); N_vec(i+1) = N_vec(i+1) + (h/2)*(u_avg^3);
        end
        N_vec(1)=0; N_vec(end)=0; R = R + N_vec; R(1)=U(1); R(end)=U(end);
        if norm(R,2) < tol, return; end
        J = K; J(1,:)=0; J(1,1)=1; J(end,:)=0; J(end,end)=1;
        dU = -J\R; U = U + dU;
    end
end