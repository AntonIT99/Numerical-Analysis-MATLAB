function [parameters,costs,accs] = model_sgd(database, layers_dims, mini_batch_size, num_epochs, val_split, learning_rate, verbose)

    %--    Arguments:
    %--    database.X_train -- dataset of shape (28x28, number of examples)
    %--    database.Y_train -- labels of shape (10, number of examples)
    %--    layers_dims -- size of layers
    %--    mini_batch_size: number of examples in mini-batch
    %--    num_epochs: number of training epochs
    %--    val_split: fraction of training examples in validation set
    %--    learning_rate: constant learning rate
    %--    verbose: true outputs cost and accuracy after each epoch
    %--
    %--    Returns:
    %--    parameters -- parameters learnt by the model. They can then be used to predict.
    %--    costs: struct containing costs.train_costs and costs.val_costs
    %--    accs: struct containing accs.train_accs and accs.val_accs


    %-- Create useful variables
    X_data = database.X_train;
    Y_data = database.Y_train;
    m = size(X_data,2);
    m_train = fix((1-val_split)*m);
    %-- Validation data
    X_val = X_data(:, m_train+1:end);
    Y_val = Y_data(:, m_train+1:end);
    %-- Training data
    X_train = X_data(:,1:m_train);
    Y_train = Y_data(:,1:m_train);
    X = X_train;
    Y = Y_train;

    train_costs = [];
    val_costs = [];
    train_accs = [];
    val_accs = [];

    %-- Parameters initialization
    parameters = initialize_parameters_deep(layers_dims);

    %-- Compute and output training accuracy after initialization
    Y_prediction_train = predict_mnist(parameters, X_train);
    h = mean(Y_prediction_train == Y_train);
    train_acc = mean(h == 1.)*100;
    disp(['Training accuracy after initialization:  ', num2str(train_acc)])
    %-- Compute and output validation accuracy after initialization
    Y_prediction_val = predict_mnist(parameters, X_val);
    h = mean(Y_prediction_val == Y_val);
    val_acc = mean(h == 1.)*100;
    disp(['Validation accuracy after initialization: ', num2str(val_acc)])
    
    

%-- mini-batch stochastic gradient descent

num_steps = fix(m_train/mini_batch_size); %steps per epoch

for k = 1:num_epochs
    % shuffle training data
    perm = randperm(m_train);
    X_shuffle = X(:,perm);
    Y_shuffle = Y(:,perm);
    fprintf('Epoch %i / %i ...\n', k, num_epochs);
    % stochastic gradient descent for epoch k
    for i=1:num_steps
        if verbose
        fprintf('Epoch %i: step %i out of %i\n', k, i, num_steps);
        end
        X_mini_batch = X_shuffle(:,i:i+mini_batch_size-1);
        Y_mini_batch = Y_shuffle(:,i:i+mini_batch_size-1);
        %-- Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        %VOTRE CODE ICI
        %
        %-- Backward propagation.
        %VOTRE CODE ICI
        %
        %-- Update parameters.
        %DECOMMENTER APRES
        %[parameters] = update_parameters(parameters, grads, learning_rate);
    end
    
        %-- Compute costs after each epoch
        [AL,~] = L_model_forward(X_train, parameters);
        train_cost = compute_cost(AL, Y_train);
        train_costs = [train_costs, train_cost];
        if verbose
        disp(['Training cost after epoch ', num2str(k), ': ', num2str(train_cost)])
        end
        [AL,~] = L_model_forward(X_val, parameters);
        val_cost = compute_cost(AL, Y_val);
        val_costs = [val_costs, val_cost];
        if verbose
        disp(['Validation cost after epoch ', num2str(k), ': ', num2str(val_cost)])
        end
        %-- Compute accuracies after each epoch
        Y_prediction_train = predict_mnist(parameters, X_train);
        h = mean(Y_prediction_train == Y_train);
        train_acc = mean(h == 1.)*100;
        if verbose
        disp(['Training accuracy after epoch ', num2str(k), ': ', num2str(train_acc)])
        end
        train_accs = [train_accs, train_acc];
        %
        Y_prediction_val = predict_mnist(parameters, X_val);
        h = mean(Y_prediction_val == Y_val);
        val_acc = mean(h == 1.)*100;
        if verbose
        disp(['Validation accuracy after epoch ', num2str(k), ': ', num2str(val_acc)])
        end
        val_accs = [val_accs, val_acc];
        
        costs.train_costs = train_costs;
        costs.val_costs = val_costs;
        accs.train_accs = train_accs;
        accs.val_accs = val_accs;
        
end

end

%---------------------------------------
%---------------------------------------
%-- Auxiliary functions


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
        parameters.W{l} = randn(layers_dims(l+1), layers_dims(l)) * sqrt(2/layers_dims(l));
        parameters.b{l} = zeros(layers_dims(l+1),1);
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
    %VOTRE CODE ICI
    %
    cost = 0; %COMMENTER APRES
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

    %-- Compute last block
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


%-- Predict
function [y_prediction] = predict(parameters, X)
    
    %-- Arguments:
    %-- X -- data set of examples you would like to label
    %-- parameters -- parameters of the trained model
    %-- 
    %-- Returns:
    %-- p -- predictions for the given dataset X

    m = size(X,2);
    n = length(parameters.W);    %-- number of layers in the neural network
    y_prediction = zeros(1,m);

    %-- Forward propagation
    [probas,~] = L_model_forward(X, parameters);


%     %-- Convert probas to 0/1 predictions
%     for i=1:m
%         if (probas(1,i) > 0.5)
%             y_prediction(1,i) = 1;
%             %y_prediction(1,i) = probas(1,i);
%         else
%             y_prediction(1,i) = 0;
%             %y_prediction(1,i) = probas(1,i);
%         end
%     end

end

%-- Predict MNIST
function [y_prediction] = predict_mnist(parameters, X)
    
    %-- Arguments:
    %-- X -- data set of examples you would like to label
    %-- parameters -- parameters of the trained model
    %-- 
    %-- Returns:
    %-- y_prediction -- predictions for the given dataset X

    m = size(X,2);
    n = length(parameters.W);    %-- number of layers in the neural network

    %-- Forward propagation
    [probas,~] = L_model_forward(X, parameters);
    %
    y_prediction = zeros(size(probas));

    %-- Convert probas to label predictions
    for i=1:m
        %VOTRE CODE ICI
        y_prediction(randi(10),i) = 1; %COMMENTER APRES
    end

end
