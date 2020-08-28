function [parameters,costs] = model(database, layers_dims, num_iterations, learning_rate, print_cost)

    %--    Arguments:
    %--    X -- dataset of shape (2, number of examples)
    %--    Y -- labels of shape (1, number of examples)
    %--    layers_dims -- size of layers
    %--    num_iterations -- Number of iterations in gradient descent loop
    %--    print_cost -- if True, print the cost every 100 iterations
    %--
    %--    Returns:
    %--    parameters -- parameters learnt by the model. They can then be used to predict.


    %-- Create usefull variables
    X = database.X_train;
    Y = database.Y_train;
    costs = [];
    X_valid = database.X_valid;
    Y_valid = database.Y_valid;
    costs_valid = [];

    %-- Création de la figure pour l'afichage en temps direct
    figure;
    
    %-- Pour l' affichage optimisé
    crb_costs = animatedline('Color','r');
    crb_costs_valid = animatedline('Color','b');
    legend('Energie Data Train','Energie Data Valid');
    xlabel('Iterations');
    xlim([0 num_iterations]);
    ylabel('Energie entropique');
    title(['Learning rate = ', num2str(learning_rate)]);


    %-- Parameters initialization
    parameters = initialize_parameters_deep(layers_dims);
    parameters_valid = initialize_parameters_deep(layers_dims);

    %-- ADAM descent
    for i=1:num_iterations

        %-- Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        [AL, caches] = L_model_forward(X, parameters);
        [AL_valid, caches_valid] = L_model_forward(X_valid, parameters_valid);

        %-- Cost function. Inputs: "AL, Y, parameters". Outputs: "cost".
        [cost] = compute_cost(AL, Y);
        [cost_valid] = compute_cost(AL_valid, Y_valid);

        %-- Backward propagation.
        [grads] = L_model_backward(AL, Y, caches);
        [grads_valid] = L_model_backward(AL_valid, Y_valid, caches_valid);
        
        %-- Initialisation des paramètres d'ADAM à la première itération.
        if(i==1)
            [parametersA] = initialize_parametersA_deep(layers_dims,grads);

        %-- Mise à jour des paramètres d'ADAM à partie de la seconde itération et à chaque itération jusqu'à la fin.
        else
            [parameters, parametersA] = update_parametersA(parameters,parametersA,grads,learning_rate);
            [parameters_valid, parametersA] = update_parametersA(parameters_valid,parametersA,grads_valid,learning_rate);
        end

        %-- Record the costs
        costs(end+1) = cost;
        costs_valid(end+1) = cost_valid;
        
        %-- Affichage optimisé en temps réel
        if (print_cost)
            addpoints(crb_costs,(i),costs(i));
            addpoints(crb_costs_valid,(i),costs_valid(i));
            drawnow
        end
        if ( print_cost && (mod(i,100)==0))
        %-- Print the cost every 100 training iterations
            disp(['TRAIN Cost after iteration ', num2str(i), ': ', num2str(cost)])
            disp(['VALID Cost after iteration ', num2str(i), ': ', num2str(cost_valid)])
            pause(0.0001);
        end
    end
    %-- ADAM descent end
    
%     %-- Gradient descent
%     for i=1:num_iterations
% 
%         %-- Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
%         [AL, caches] = L_model_forward(X, parameters);
%         [AL_valid, caches_valid] = L_model_forward(X_valid, parameters_valid);
% 
%         %-- Cost function. Inputs: "AL, Y, parameters". Outputs: "cost".
%         [cost] = compute_cost(AL, Y);
%         [cost_valid] = compute_cost(AL_valid, Y_valid);
% 
%         %-- Backward propagation.
%         [grads] = L_model_backward(AL, Y, caches);
%         [grads_valid] = L_model_backward(AL_valid, Y_valid, caches_valid);
% 
%         %-- Update parameters.
%         [parameters] = update_parameters(parameters, grads, learning_rate);
%         [parameters_valid] = update_parameters(parameters_valid, grads_valid, learning_rate);
% 
%         %-- Record the costs
% %         if ( mod(i,100) == 0)
% %             costs(end+1) = cost;
% %         end
%         costs(end+1) = cost;
%         costs_valid(end+1) = cost_valid;
% 
%         %-- Affichage en temps réel sur la figure (pas optimisé)       
% %         if(print_cost)
% %             plot(costs,'r'); %/ affichage du coût de la base d'entrainement
% %             hold on; 
% %             plot(costs_valid,'b'); %/ affichage du coût de la base de validation
% %             hold on; 
% %             legend('Energie Data Train','Energie Data Valid');
% %             pause(0.0001);
% %  
% %             if(i==1) %/ permet de formater les caractéristiques de la courbe dans la figure
% %                 xlabel('itérations');
% %                 xlim([0 num_iterations]);
% %                 ylabel('Energie entropique');
% %                 title(['Learning rate =', num2str(learning_rate)]);
% %             end
% %         end
%         
%         %-- Affichage optimisé en temps réel
%         if (print_cost)
%             addpoints(crb_costs,(i),costs(i));
%             addpoints(crb_costs_valid,(i),costs_valid(i));
%             drawnow
%         end
% 
%         %-- Print the cost every 100 training iterations
%         if ( print_cost && (mod(i,100)==0) )
%             %disp(['Cost after iteration ', num2str(i), ': ', num2str(cost)])
%             disp(['TRAIN Cost after iteration ', num2str(i), ': ', num2str(cost)])
%             disp(['VALID Cost after iteration ', num2str(i), ': ', num2str(cost_valid)])
%             pause(0.0001);
%         end
%     end
%     %-- Gradient descent end

    %-- Affichage du coût à la fin de l'exécution du programme
     %if (print_cost)
         %plot(costs,'r');
         %hold on;
         %plot(costs_valid,'b');
         %xlabel('itérations');
         %ylabel('Energie entropique');
         %title(['Learning rate =', num2str(learning_rate)]);
     %end
end


%---------------------------------------
%---------------------------------------
%-- Auxillary functions


function [parameters] = initialize_parameters_deep(layers_dims)

    %--    Arguments:
    %--    layer_dims -- array containing the dimensions of each layer in our network
    %--
    %--    Returns:
    %--    parameters -- matlab structure containing parameters "W1", "b1", ..., "WL", "bL":
    %--                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
    %--                    bl -- bias vector of shape (layer_dims[l], 1)
    %--    Wl is retrieved thanks to the following command: parameters.W{l}

    L = length(layers_dims);      %-- number of layers in the network

    for l=1:(L-1)
        % parameters.W{l} = randn(layers_dims(l+1), layers_dims(l)) / (1.5*sqrt(layers_dims(l)) );
        parameters.W{l} = randn(layers_dims(l+1), layers_dims(l)) * sqrt(2/layers_dims(l));
        parameters.b{l} = zeros(layers_dims(l+1),1);
    end

end


function [parametersA] = initialize_parametersA_deep(layers_dims,grads)

    L = length(layers_dims);
    parametersA.mt = grads;
    parametersA.vt = grads;

    for l=1:L-1
        parametersA.vt.dW{l} =(grads.dW{l}).^2;
        parametersA.vt.db{l} =(grads.db{l}).^2;
    end 
end

function [parameters, parametersA] = update_parametersA(parameters,parametersA,grads,learning_rate)
 
    %-- Initialisation des paramètres utiles et ceux nécessaires pour mettre à jour mt et vt
    L = length(parameters.W);        
    delta = 10^(-10);
    b1 = 0.6;
    b2 = 0.7;
    

    %-- Méthode ADAM
    for l=1:L 
        %-- Mise à jour des poids W
        parametersA.mt.dW{l} = b1*parametersA.mt.dW{l} + (1-b1)*grads.dW{l};
        parametersA.vt.dW{l} = b2*parametersA.vt.dW{l} + (1-b2)*(grads.dW{l}.^2);
        parameters.W{l} = parameters.W{l} - learning_rate *(1/(1-b1)*parametersA.mt.dW{l}./(sqrt(parametersA.vt.dW{l})/(1-b2))+delta);
        
        %-- Mise à jour des biais b
        parametersA.mt.db{l} = b1*parametersA.mt.db{l} + (1-b1)*grads.db{l};
        parametersA.vt.db{l} = b2*parametersA.vt.db{l} + (1-b2)*(grads.db{l}.^2);
        parameters.b{l} = parameters.b{l} - learning_rate *(1/(1-b1)*parametersA.mt.db{l}./(sqrt(parametersA.vt.db{l})/(1-b2))+delta);
    end
end



%-- Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
function [AL, caches] = L_model_forward(X, parameters)

    %-- Arguments:
    %-- X -- data, array of shape (input size, number of examples)
    %-- parameters -- output of initialize_parameters_deep()
    %--
    %-- Returns:
    %-- AL -- last post-activation value
    %-- caches -- list of caches containing:
    %--             every cache of linear_relu_forward() (there are L-1 of them, indexed from 1 to L-1)
    %--             the cache of linear_sigmoid_forward() (there is one, indexed L)

    A = X;
    L = length(parameters.W);     %-- number of layers in the neural network
    caches = cell(1,L);

    %-- Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    for l=1:(L-1)
        A_prev = A;
        [A, cache] = linear_activation_forward(A_prev, parameters.W{l}, parameters.b{l}, 'relu');
        caches{l} = cache;
    end

    %-- Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
    [AL, cache] = linear_activation_forward(A, parameters.W{L}, parameters.b{L}, 'sigmoid');
    caches{L} = cache;

end


%--Implement the forward propagation for the LINEAR->ACTIVATION layer
function [A, cache] = linear_activation_forward(A_prev, W, b, activation)

    %-- Arguments
    %-- A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    %-- W -- weights matrix of shape (size of current layer, size of previous layer)
    %-- b -- bias vector of shape (size of the current layer, 1)
    %-- activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
    %--
    %-- Returns
    %-- A -- the output of the activation function, also called the post-activation value
    %-- cache -- a matlab structure containing "W", "b", "Z" and "A";
    %--          stored for computing the backward pass efficiently

    [Z, cache] = linear_forward(A_prev, W, b);

    if ( strcmp(activation,'sigmoid')==1 )
        [A] = sigmoid(Z);
    end

    if ( strcmp(activation,'relu')==1 )
        [A] = relu(Z);
    end

end


%-- Implement the linear part of a layer's forward propagation.
function [Z, cache] = linear_forward(A, W, b)

    %-- Arguments
    %-- A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    %-- W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    %-- b -- bias vector, numpy array of shape (size of the current layer, 1)
    %--
    %-- Returns:
    %-- Z -- the input of the activation function, also called pre-activation parameter
    %-- cache -- a matlab structure containing "W", "b", "Z" and "A" ; stored for computing the backward pass efficiently

    %Z = W*A + b;
    
    
    WA = W*A;
    Z = WA + b*ones(1,size(WA,2));
    
    cache.W = W;
    cache.b = b;
    cache.Z = Z;
    cache.A = A;

end


%-- Implements the sigmoid function
function [A] = sigmoid(Z)

    %-- Arguments:
    %-- Z -- array of any shape
    %-- Returns:
    %-- A -- output of sigmoid(z), same shape as Z

    A = 1./(1+exp(-Z));

end


%-- Implement the RELU function.
function [A] = relu(Z)

    %-- Arguments
    %-- Z -- Output of the linear layer, of any shape
    %--
    %-- Returns:
    %-- A -- Post-activation parameter, of the same shape as Z

    A = max(0,Z);

end


%-- Implement the cross-entropy cost function
function [cost] = compute_cost(AL, Y)

    %--    Arguments:
    %--    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    %--    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)
    %--
    %--    Returns:
    %--    cost -- cross-entropy cost

    m = size(Y,2);

    %-- Compute the cross-entropy cost
    cost = -( (Y*(log(AL+eps))') + ( (1-Y)*(log(1-AL+eps))') ) / m;    %-- cost

end


%-- Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
function [grads] = L_model_backward(AL, Y, caches)

    %-- Arguments:
    %-- AL -- probability vector, output of the forward propagation (L_model_forward())
    %-- Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    %-- caches -- list of caches containing:
    %--             every cache of linear_activation_forward() with "relu" (there are (L-1) or them, indexes from 1 to L-1)
    %--             the cache of linear_activation_forward() with "sigmoid" (there is one, index L)
    %--
    %-- Returns:
    %-- grads -- A matlab structure containing the gradients
    %--          grads.dA{l} = ...
    %--          grads.dW{l} = ...
    %--          grads.db{l} = ...

    L = length(caches);  %-- the number of layers

    %-- Initializing the backpropagation
    dAL = - ( (Y./(AL+eps)) - (1-Y)./(1-AL + eps) );

    %-- Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "AL, Y, caches". Outputs: "grads["dAL"], grads["dWL"], grads["dbL"]
  %-- Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "AL, Y, caches". Outputs: "grads["dAL"], grads["dWL"], grads["dbL"]
    if (L==1)
    [~, grads.dW{L}, grads.db{L}] = linear_activation_backward(dAL, caches{L}, 'sigmoid');   
    else
    [grads.dA{L-1}, grads.dW{L}, grads.db{L}] = linear_activation_backward(dAL, caches{L}, 'sigmoid');
    %-- lth layer: (RELU -> LINEAR) gradients.
    for l=(L-1):-1:2
        [grads.dA{l-1}, grads.dW{l}, grads.db{l}] = linear_activation_backward(grads.dA{l}, caches{l}, 'relu');
    end

    %-- Compute last bloc
    [~, grads.dW{1}, grads.db{1}] = linear_activation_backward(grads.dA{1}, caches{1}, 'relu');
    end

end


%-- Implement the backward propagation for the LINEAR->ACTIVATION layer.
function [dA_prev, dW, db] = linear_activation_backward(dA, cache, activation)

    %-- Arguments:
    %-- dA -- post-activation gradient for current layer l
    %-- cache -- (linear_cache, activation_cache) values we stored for computing backward propagation efficiently
    %-- activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
    %--
    %-- Returns:
    %-- dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    %-- dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    %-- db -- Gradient of the cost with respect to b (current layer l), same shape as b


    if ( strcmp(activation,'relu') == 1 )
        dZ = relu_backward(dA, cache);
        [dA_prev, dW, db] = linear_backward(dZ, cache);
    end
    if ( strcmp(activation,'sigmoid') == 1 )
        dZ = sigmoid_backward(dA, cache);
        [dA_prev, dW, db] = linear_backward(dZ, cache);
    end

end


%-- Implement the backward propagation for a single RELU unit.
function [dZ] = relu_backward(dA, cache)

    %-- Arguments
    %-- dA -- post-activation gradient, of any shape
    %-- cache -- 'Z' where we store for computing backward propagation efficiently
    %--
    %-- Returns:
    %-- dZ -- Gradient of the cost with respect to Z

    Z = cache.Z;
    dZ = dA;

    %-- When z <= 0, you should set dz to 0 as well.
    dZ(Z <= 0) = 0;

end


%-- Implement the backward propagation for a single SIGMOID unit.
function [dZ] = sigmoid_backward(dA, cache)

    %-- Arguments:
    %-- dA -- post-activation gradient, of any shape
    %-- cache -- 'Z' where we store for computing backward propagation efficiently
    %--
    %-- Returns:
    %-- dZ -- Gradient of the cost with respect to Z

    Z = cache.Z;

    s = 1./(1+exp(-Z));
    dZ = dA .* s .* (1-s);

end


%-- Implement the linear portion of backward propagation for a single layer (layer l)
function [dA_prev, dW, db] = linear_backward(dZ, cache)

    %-- Arguments:
    %-- dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    %-- cache -- (A_prev, W, b) values coming from the forward propagation in the current layer
    %--
    %-- Returns:
    %-- dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    %-- dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    %-- db -- Gradient of the cost with respect to b (current layer l), same shape as b

    A_prev = cache.A;
    W = cache.W;
    m = size(A_prev,2);

    dW = 1/m * (dZ*(A_prev'));
    db = 1/m * sum(dZ, 2);
    dA_prev = (W')*dZ;

end


%-- Update parameters using gradient descent
function parameters = update_parameters(parameters, grads, learning_rate)

    %-- Arguments:
    %-- parameters -- matlab structure containing your parameters
    %-- grads -- matlab structure containing your gradients, output of L_model_backward
    %--
    %-- Returns:
    %-- parameters -- matlab structure containing your updated parameters
    %--               parameters.W{l} = ...
    %--               parameters.b{l} = ...

    L = length(parameters.W);

    %-- Update rule for each parameter. Use a for loop.
    for l=1:L
        parameters.W{l} = parameters.W{l} - learning_rate * grads.dW{l};
        parameters.b{l} = parameters.b{l} - learning_rate * grads.db{l};
    end

end
