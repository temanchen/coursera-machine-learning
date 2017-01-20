function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% feed forward
a1 = [ones(m, 1) X];
a2 = sigmoid(a1 * Theta1');

a2 = [ones(m, 1) a2];
a3 = sigmoid(a2 * Theta2');

% =========================================================================

D3 = zeros(num_labels, 1);
D2 = zeros(hidden_layer_size, 1);

G2 = zeros(num_labels, hidden_layer_size + 1);
G1 = zeros(hidden_layer_size, input_layer_size + 1);

I = eye(num_labels);

for i = 1:m
    y_i = I(:, y(i));
    h_i = a3(i, :)';

    % accumulate cost
    J -= (y_i' * log(h_i) + (1 - y_i') * log(1 - h_i)) / m;

    % compute delta
    D3 = h_i - y_i;
    D2 = (a2(i, 2:end) .* (1 - a2(i, 2:end)))' .* (Theta2(:, 2:end)' * D3);

    % accumulate gradient
    G1 += D2 * a1(i, :);
    G2 += D3 * a2(i, :);
end

% regularization
theta1_bias = Theta1(:, 1);
theta2_bias = Theta2(:, 1);
J += lambda / (2 * m) * (nn_params' * nn_params - theta1_bias' * theta1_bias - theta2_bias' * theta2_bias);

Theta1_grad = G1 / m + lambda / m * Theta1;
Theta1_grad(:, 1) -= lambda / m * Theta1(:, 1);

Theta2_grad = G2 / m + lambda / m * Theta2;
Theta2_grad(:, 1) -= lambda / m * Theta2(:, 1);

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];
