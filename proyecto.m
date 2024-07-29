clc;clear;close all;
%% Directorios de las imágenes de entrenamiento
dir_hombres = 'C:/Users/EstJustinXavierRosil/OneDrive - Universidad Politecnica Salesiana\Documentos/Universidad/Periodo 64/Inteligencia articial/Parcial2/proyecto/archive/Training/male';
dir_mujeres = 'C:/Users/EstJustinXavierRosil/OneDrive - Universidad Politecnica Salesiana\Documentos/Universidad/Periodo 64/Inteligencia articial/Parcial2/proyecto/archive/Training/female';

%% Listar archivos de imágenes
archivos_hombres = dir(fullfile(dir_hombres, '*.jpg'));
archivos_mujeres = dir(fullfile(dir_mujeres, '*.jpg'));

%% Seleccionar las primeras 100 imágenes
archivos_hombres = archivos_hombres(1:100);
archivos_mujeres = archivos_mujeres(1:100);

%% Conversión a escala de grises y redimensionamiento
num_imagenes = 100;
alto_img = 90;
ancho_img = 60;
tam_img = [alto_img, ancho_img];

Xtrain = zeros(num_imagenes * 2, alto_img * ancho_img);
ytrain = [ones(num_imagenes, 1); -ones(num_imagenes, 1)];

for i = 1:num_imagenes
    % Procesar imágenes de hombres
    img = imread(fullfile(dir_hombres, archivos_hombres(i).name));
    img_gris = rgb2gray(img);
    img_redimensionada = imresize(img_gris, tam_img);
    Xtrain(i, :) = img_redimensionada(:)';

    % Procesar imágenes de mujeres
    img = imread(fullfile(dir_mujeres, archivos_mujeres(i).name));
    img_gris = rgb2gray(img);
    img_redimensionada = imresize(img_gris, tam_img);
    Xtrain(num_imagenes + i, :) = img_redimensionada(:)';
end

%% Vector de pesos inicial
w_inicial = zeros(1, alto_img * ancho_img);

%% Función perceptronLearning
[w_final, W_values] = perceptronLearning(Xtrain, ytrain, w_inicial);

%% Directorios de las imágenes de validación
dir_val_hombres = 'C:/Users/EstJustinXavierRosil/OneDrive - Universidad Politecnica Salesiana/Documentos/Universidad/Periodo 64/Inteligencia articial/Parcial2/proyecto/archive/Validation/male';
dir_val_mujeres = 'C:/Users/EstJustinXavierRosil/OneDrive - Universidad Politecnica Salesiana/Documentos/Universidad/Periodo 64/Inteligencia articial/Parcial2/proyecto/archive/Validation/female';

%% Listar archivos de imágenes
archivos_val_hombres = dir(fullfile(dir_val_hombres, '*.jpg'));
archivos_val_mujeres = dir(fullfile(dir_val_mujeres, '*.jpg'));

%% Seleccionar las primeras imágenes
archivos_val_hombres = archivos_val_hombres(1:20);
archivos_val_mujeres = archivos_val_mujeres(1:24);

num_val_imagenes = 44;
xtest = zeros(num_val_imagenes, alto_img * ancho_img);
yetiquetas_val = [ones(20, 1); -ones(24, 1)];

for i = 1:20
    % Procesar imágenes de hombres
    img = imread(fullfile(dir_val_hombres, archivos_val_hombres(i).name));
    img_gris = rgb2gray(img);
    img_redimensionada = imresize(img_gris, tam_img);
    xtest(i, :) = img_redimensionada(:)';
end

for i = 1:24
    % Procesar imágenes de mujeres
    img = imread(fullfile(dir_val_mujeres, archivos_val_mujeres(i).name));
    img_gris = rgb2gray(img);
    img_redimensionada = imresize(img_gris, tam_img);
    xtest(20 + i, :) = img_redimensionada(:)';
end

%% Función perceptronOutput
y_hat = perceptronOutput(xtest, w_final);

%% Definir etiquetas de predicción
etiquetas_pred = cell(num_val_imagenes, 1);
etiquetas_pred(y_hat > 0) = {'Hombre'};
etiquetas_pred(y_hat <= 0) = {'Mujer'};

%% Selección aleatoria de 20 imágenes
idx = randperm(num_val_imagenes, 20);

%% Visualización
figure;
for i = 1:20
    subplot(4, 5, i);
    img = reshape(xtest(idx(i), :), tam_img);
    imshow(img, []);
    title(['Pred: ', etiquetas_pred{idx(i)}]);
end

%% Función perceptronLearning
function [w_final, W_values] = perceptronLearning(Xtrain, ytrain, w_inicial)
    % Inicialización
    [num_samples, num_features] = size(Xtrain);
    w = w_inicial;
    W_values = w;
    
    % Parámetros
    tasa_aprendizaje = 0.1;
    max_iter = 1000;
    iter = 0;
    error = Inf;
    
    % Ciclo de aprendizaje
    while error > 0.1 && iter < max_iter
        error = 0;
        for i = 1:num_samples
            if ytrain(i) * (Xtrain(i, :) * w') <= 0
                w = w + tasa_aprendizaje * ytrain(i) * Xtrain(i, :);
                error = error + 1;
            end
        end
        W_values = [W_values; w];
        iter = iter + 1;
    end
    w_final = w;
end

%% Función perceptronOutput
function y_hat = perceptronOutput(X, w)
    num_samples = size(X, 1);
    y_hat = zeros(num_samples, 1);
    for i = 1:num_samples
        y_hat(i) = sign(X(i, :) * w');
    end
end
