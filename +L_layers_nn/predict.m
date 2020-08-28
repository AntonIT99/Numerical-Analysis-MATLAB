%-- This function is used to predict the results of a  L-layer neural network.
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


    %-- Convert probas to 0/1 predictions
    for i=1:m
        if (probas(1,i) > 0.5)
            y_prediction(1,i) = 1;
        else
            y_prediction(1,i) = 0;
        end
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
    %--             every cache of linear_relu_forward() (there are L-1 of them, indexed from 0 to L-2)
    %--             the cache of linear_sigmoid_forward() (there is one, indexed L-1)

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
    %-- W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    %-- b -- bias vector, numpy array of shape (size of the current layer, 1)
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

    Z = W*A + b;
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


%-- Implement the hyperbolic tangent function
function [A] = tanh(Z)
    
    %-- Arguments: 
    %-- Z --> array of any size.
    %-- Return: 
    %-- A -- output of tanh(z), same shape as Z

    A = (exp(z)-exp(-z))./(exp(z)+exp(-z));
    
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



