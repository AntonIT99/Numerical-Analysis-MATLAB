clear all;
close all;
more off;
clc;
addpath('../');

%-- Load 60000 images of MNIST database in X_train and corresponding labels
% in Y_train
filename = '../data/mnist_train_data.mat';
load(filename);
% Rename data
X_data = double(X_train);
Y_data = double(Y_train);
% Define reduced training set
num_examples = 6000;
X_train = X_data(:,1:num_examples);
Y_train = Y_data(:,1:num_examples);
database.X_train = X_train;
database.Y_train = Y_train;
val_split = 0.2;
% %-- Display training database examples
% figure;
% for i=1:25
%     im = reshape(X_train(:,i),28,28);
%     [~, label] = max(Y_train(:,i)); label = label - 1;
%     subplot(5,5,i); imagesc(im); title(['label: ', num2str(label)]);
% end


% Initialize random generator 
rng('default');
rng(1);
% Define parameters for SGD
num_epochs = 1; %10,100,200
mini_batch_size = 10;
learning_rate = 0.1;
verbose = true;
%-- Build a model with a n_hidden-dimensional hidden layer
n_hidden = 100;
nX = size(X_train,1);
nY = size(Y_train,1);
layers_dims = [nX, n_hidden, nY];
[parameters,costs,accs] = L_layers_nn.model_sgd_a_completer(database, layers_dims, mini_batch_size, num_epochs, val_split, learning_rate, verbose);
% Plot costs and accuracies
train_costs = costs.train_costs;
val_costs = costs.val_costs;
figure; plot(train_costs); hold on;
plot(val_costs); legend('training cost', 'validation cost', 'Location', 'northeast');
train_accs = accs.train_accs;
val_accs = accs.val_accs;
figure; plot(train_accs); hold on;
plot(val_accs); legend('train accs', 'val accs', 'Location', 'northwest');

