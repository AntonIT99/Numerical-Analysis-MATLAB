close all;
clear all;
clc;

load('database.mat');

%--Initialisation--%
img1=imgs_2CH{72}; %on charge l'image A2C d'indice 1
img2=imgs_4CH{72}; %on charge l'image A4C d'indice 1

%--Affichage--%

figure;
imagesc(img1); %affichage de l'image
axis image; %rescaling 
colormap(gray); %nuances de gris
colorbar; %échelle des couleurs

figure;
imagesc(img2);
axis image;
colormap(gray);
colorbar;

%%%%--sépartation de la base de donnée en 3 groupes--%%%%

liste=randperm(500); %création d'un vecteur de taille 500 comportant les valeurs 1 à 500 dans le désordre

%X_name : images sous forme vectorielle grâce à la fonction img(:)
%Y_name : contient les valeurs 0 ou 1 pour respectivement A2C ou A4C


%--Base de validation : 800 images au hasard (400 A2C et 400 A4C)

% for i=1:400
% img=imgs_2CH{liste(i)}; %prend l'image A2C du patient i
% database.X_train{i}=img(:);
% database.Y_train(i)=0; %A2C => y=0 d'après l'énoncé
% 
% img=imgs_4CH{liste(i)}; %prend l'image A4C du patient i
% database.X_train{400+i}=img(:);
% database.Y_train(400+i)=1; %A4C => y=1 d'après l'énoncé
% end
% database.X_train=cell2mat(database.X_train); %transformation en matrice
% 
% %--Base de validation : 100 images au hasard parmi celles restantes (50 A2C et 50 A4C)
% for i=401:450
% img=imgs_2CH{liste(i)};
% database.X_valid{i-400}=img(:);
% database.Y_valid(i-400)=0;
% 
% img=imgs_4CH{liste(i)};
% database.X_valid{i-350}=img(:);
% database.Y_valid(i-350)=1;
% end
% database.X_valid=cell2mat(database.X_valid);
% 
% %--Base de test : 100 images au hasard parmi celles restantes (50 A2C et 50 A4C)
% for i=451:500
% img=imgs_2CH{liste(i)};
% database.X_test{i-450}=img(:);
% database.Y_test(i-450)=0;
% 
% img=imgs_4CH{liste(i)};
% database.X_test{i-400}=img(:);
% database.Y_test(i-400)=1;
% end
% database.X_test=cell2mat(database.X_test);
% 
% 
% 
% %%%%--Création du modèle de réseaux de neurones à 1 couche--%%%%
% 
% num_iterations=700;
% learning_rate=0.001;
% print_cost=true;
% nX=size(database.X_train,1); %nombre de neurones de la couche d’entrée = dimension de X_train
% layers_dims=[nX, 4, 4, 1]; %couches
% [parameters,costs] = L_layers_nn.model(database, layers_dims, num_iterations, learning_rate, print_cost);
% 
% %--Initialisation des variables
% X_train=database.X_train;
% Y_train=database.Y_train;
% X_test=database.X_test;
% Y_test=database.Y_test;
% X_valid=database.X_valid;
% Y_valid=database.Y_valid;
% 
% %--Calculdelaprécision
% Y_prediction_train=L_layers_nn.predict(parameters, X_train);
% Y_prediction_test=L_layers_nn.predict(parameters, X_test);
% Y_prediction_valid=L_layers_nn.predict(parameters, X_valid);
% 
% %--Affichage de la précision des groupes
% disp(['train accuracy:', num2str(100 - mean(abs(Y_prediction_train - Y_train)) * 100),'%']);
% disp(['test accuracy:', num2str(100 - mean(abs(Y_prediction_test - Y_test)) * 100),'%']);
% disp(['valid accuracy:', num2str(100 - mean(abs(Y_prediction_valid - Y_valid)) * 100),'%']);
% 
% 
