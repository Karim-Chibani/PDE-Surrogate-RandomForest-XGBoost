% =========================================================================
% SCRIPT : Résolution d'une EDP non linéaire par la MEF P1 + Newton-Raphson
% Équation : -u'' + u^3 = sin(pi*x) sur [0, 1] avec u(0) = u(1) = 0
% =========================================================================
clear; clc;

% 1. PARAMÈTRES DU MAILLAGE
N = 40;                     % Nombre d'éléments
x = linspace(0, 1, N + 1)';   % : Vecteur colonne des nœuds
h = 1 / N;                   % Pas de maillage (constant ici)

% 2. DÉFINITION DES FONCTIONS
f = @(v) sin(pi * v);        % Terme source f(x)

% 3. INITIALISATION DU VECTEUR SOLUTION (Newton-Raphson)
U = zeros(N + 1, 1);         % Premier choix initial u_0 = 0

% Paramètres de convergence
tol = 1e-8;                  % Tolérance du résidu
max_iter = 30;               % Nombre maximal d'itérations
converged = false;

fprintf('Début de l''algorithme de Newton-Raphson...\n');
fprintf('Iter \t Norme du Résidu\n');
fprintf('-------------------------\n');

% 4. BOUCLE PRINCIPALE DE NEWTON-RAPHSON
for iter = 1:max_iter
    
    % --- ASSEMBLAGE DU RÉSIDU R(U) = K*U + N(U) - F ---
    
    % a. Partie linéaire : Matrice de rigidité K
    K = zeros(N + 1, N + 1);
    for i = 1:N
        K_elem = [1/h, -1/h; -1/h, 1/h];
        K(i:i+1, i:i+1) = K(i:i+1, i:i+1) + K_elem;
    end
    
    % b. Partie non linéaire : Vecteur N(U) correspondant au terme u^3
    % Approximation par la méthode des trapèzes sur chaque élément
    N_vec = zeros(N + 1, 1);
    for i = 1:N
        u_avg = 0.5 * (U(i) + U(i+1)); % Valeur moyenne locale de u
        N_elem = [ (h/2)*(u_avg^3) ; (h/2)*(u_avg^3) ];
        N_vec(i:i+1) = N_vec(i:i+1) + N_elem;
    end
    
    % c. Vecteur second membre F (Terme source f(x))
    F = zeros(N + 1, 1);
    for i = 1:N
        F_elem = [ (h/2)*f(x(i)) ; (h/2)*f(x(i+1)) ];
        F(i:i+1) = F(i:i+1) + F_elem;
    end
    
    % Construction du Résidu Global
    R = K * U + N_vec - F;
    
    % --- ASSEMBLAGE DE LA MATRICE JACOBIENNE J(U) = K + M_nonlin ---
    J = K;
    for i = 1:N
        u_avg = 0.5 * (U(i) + U(i+1));
        % Dérivée de u^3 est 3*u^2
        J_elem = [ (h/2)*3*u_avg^2, 0 ; 0, (h/2)*3*u_avg^2 ];
        J(i:i+1, i:i+1) = J(i:i+1, i:i+1) + J_elem;
    end
    
    % --- APPLICATION DES CONDITIONS AUX LIMITES (Dirichlet Homogènes) ---
    % On force le résidu à 0 aux extrémités
    R(1) = U(1) - 0;
    R(end) = U(end) - 0;
    
    % Modification de la Jacobienne pour bloquer les valeurs aux bords
    J(1, :) = 0;     J(1, 1) = 1;
    J(end, :) = 0;   J(end, end) = 1;
    
    % --- VÉRIFICATION DE LA CONVERGENCE ---
    res_norm = norm(R, 2);
    fprintf('%d \t\t %e\n', iter, res_norm);
    
    if res_norm < tol
        converged = true;
        fprintf('-------------------------\n');
        fprintf('Convergence atteinte à l''itération %d.\n', iter);
        break;
    end
    
    % --- MISE À JOUR : Résolution du système linéaire J * dU = -R ---
    dU = - J \ R;
    U = U + dU;
end

if ~converged
    warning('L''algorithme n''a pas convergé après %d itérations.', max_iter);
end

% =========================================================================
% 5. VISUALISATION DE LA SOLUTION TRACÉE (Jury-Ready)
% =========================================================================
figure;
plot(x, U, 'b-o', 'LineWidth', 2, 'MarkerSize', 4);
grid on;
title('Solution $u(x)$ obtenue par la méthode des Éléments Finis $P_1$', 'Interpreter', 'latex');
xlabel('Espace $x$', 'Interpreter', 'latex');
ylabel('Solution $u(x)$', 'Interpreter', 'latex');