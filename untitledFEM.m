% =========================================================================
% MATLAB : Génération de données pour MULTIPLES fonctions f(x)
% =========================================================================
clear; clc;

N = 100; % Nombre d'éléments (101 points)
x_nodes = linspace(0, 1, N + 1)'; 

% Nombre de fonctions f à générer pour l'entraînement
nb_fonctions = 50; 

filename = 'pde_data_multi.csv';
fileID = fopen(filename, 'w');

fprintf('Génération des données pour %d fonctions f(x)...\n', nb_fonctions);

for k = 1:nb_fonctions
    % Génération d'une fonction f(x) aléatoire (Combinaison de sinus/polynômes)
    % Chaque itération k aura une fonction f différente
    a = -5 + 10*rand(); % Amplitude aléatoire entre -5 et 5
    b = -2 + 4*rand();
    f_fun = @(x) a*sin(pi*x) + b*x.*(1-x); 
    
    % Calcul des features
    f_values = f_fun(x_nodes);
    f_mean = mean(f_values);
    
    % RESOLUTION FEM (Ici, mettez l'appel exact de votre code Newton/U)
    % [Simulé ici par une boucle Newton classique pour que le script tourne direct]
    U = zeros(N+1, 1); 
    tol = 1e-6; max_iter = 30;
    K = zeros(N+1, N+1);
    for i = 1:N
        h = x_nodes(i+1) - x_nodes(i);
        K(i:i+1, i:i+1) = K(i:i+1, i:i+1) + [1/h, -1/h; -1/h, 1/h];
    end
    F = zeros(N+1, 1);
    for i = 1:N
        h = x_nodes(i+1) - x_nodes(i);
        F(i) = F(i) + (h/2)*f_values(i); F(i+1) = F(i+1) + (h/2)*f_values(i+1);
    end
    for iter = 1:max_iter
        R = K*U - F;
        for i = 1:N
            h = x_nodes(i+1)-x_nodes(i); u_avg = 0.5*(U(i)+U(i+1));
            R(i) = R(i)+(h/2)*u_avg^3; R(i+1) = R(i+1)+(h/2)*u_avg^3;
        end
        R(1) = U(1); R(end) = U(end);
        if norm(R,2) < tol, break; end
        J = K;
        for i = 1:N
            h = x_nodes(i+1)-x_nodes(i); u_avg = 0.5*(U(i)+U(i+1));
            J(i:i+1, i:i+1) = J(i:i+1, i:i+1) + [(h/2)*3*u_avg^2, 0; 0, (h/2)*3*u_avg^2];
        end
        J(1,:) = 0; J(1,1) = 1; J(end,:) = 0; J(end,end) = 1;
        U = U - J\R;
    end
    
    % Écriture dans le fichier CSV
    for i = 1:length(x_nodes)
        fprintf(fileID, '%f,%f,%f,%f\n', x_nodes(i), f_values(i), f_mean, U(i));
    end
end

fclose(fileID);
fprintf('Succès ! Le fichier "%s" contient %d lignes.\n', filename, nb_fonctions*(N+1));