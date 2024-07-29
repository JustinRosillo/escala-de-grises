clc; clear; close all;

%% Directorios de las imágenes de entrenamiento
dir_hombres = 'C:/Users/EstJustinXavierRosil/OneDrive - Universidad Politecnica Salesiana/Documentos/Universidad/Periodo 64/Inteligencia articial/Parcial2/proyecto/archive/Training/male';
dir_mujeres = 'C:/Users/EstJustinXavierRosil/OneDrive - Universidad Politecnica Salesiana/Documentos/Universidad/Periodo 64/Inteligencia articial/Parcial2/proyecto/archive/Training/female';

%% Listar archivos de imágenes
archivos_hombres = dir(fullfile(dir_hombres, '*.jpg'));
archivos_mujeres = dir(fullfile(dir_mujeres, '*.jpg'));

%% Seleccionar la segunda y penúltima imagen de cada clase
% Para hombres
img_hombre_segunda = imread(fullfile(dir_hombres, archivos_hombres(2).name));
img_hombre_penultima = imread(fullfile(dir_hombres, archivos_hombres(end-1).name));

% Para mujeres
img_mujer_segunda = imread(fullfile(dir_mujeres, archivos_mujeres(2).name));
img_mujer_penultima = imread(fullfile(dir_mujeres, archivos_mujeres(end-1).name));

%% Convertir imágenes a escala de grises
img_hombre_segunda_gray = rgb2gray(img_hombre_segunda);
img_hombre_penultima_gray = rgb2gray(img_hombre_penultima);
img_mujer_segunda_gray = rgb2gray(img_mujer_segunda);
img_mujer_penultima_gray = rgb2gray(img_mujer_segunda);

%% Redimensionar imágenes a 90x60 píxeles
tam_img = [90, 60];
img_hombre_segunda_resized = imresize(img_hombre_segunda_gray, tam_img);
img_hombre_penultima_resized = imresize(img_hombre_penultima_gray, tam_img);
img_mujer_segunda_resized = imresize(img_mujer_segunda_gray, tam_img);
img_mujer_penultima_resized = imresize(img_mujer_penultima_gray, tam_img);

%% Crear una figura para mostrar las imágenes con dimensiones
figure;
subplot(2, 2, 1);
imshow(img_hombre_segunda_resized);
title('Hombre - Segunda Imagen');
addDimensionsBar(img_hombre_segunda_resized);

subplot(2, 2, 2);
imshow(img_hombre_penultima_resized);
title('Hombre - Penúltima Imagen');
addDimensionsBar(img_hombre_penultima_resized);

subplot(2, 2, 3);
imshow(img_mujer_segunda_resized);
title('Mujer - Segunda Imagen');
addDimensionsBar(img_mujer_segunda_resized);

subplot(2, 2, 4);
imshow(img_mujer_penultima_resized);
title('Mujer - Penúltima Imagen');
addDimensionsBar(img_mujer_penultima_resized);

%% Función para agregar las marcas de dimensiones cada 20 píxeles
function addDimensionsBar(img)
    hold on;
    [alto, ancho] = size(img);
    % Marcas de 20 píxeles en la barra vertical (altura), excluyendo 0
    for y = 20:20:alto
        text(-15, y, num2str(y), 'VerticalAlignment', 'middle', 'HorizontalAlignment', 'right', 'FontSize', 8, 'Color', 'k');
    end
    
    % Marcas de 20 píxeles en la barra horizontal (ancho), excluyendo 0
    for x = 20:20:ancho
        text(x, alto + 15, num2str(x), 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'center', 'FontSize', 8, 'Color', 'k');
    end
    
    % Agregar la barra vertical (altura) solo con marcas
    line([-10 -10], [0 alto], 'Color', 'k', 'LineWidth', 1);

    % Agregar la barra horizontal (ancho) solo con marcas
    line([0 ancho], [alto + 10 alto + 10], 'Color', 'k', 'LineWidth', 1);
    
    hold off;
end
