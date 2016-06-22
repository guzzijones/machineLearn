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
size(Theta1);
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
size(Theta2);
% Setup some useful variables
m = size(X, 1);
%forward 1
#prepend oneszeros
%fprintf('size X 1...\n');
size(X,1);
predictions = zeros(size(X,1),1);
size(predictions);
X = [ones(m, 1) X];

%for i= i:rows(X)
%fprintf('in for..\n')
  %pause
 % predictionsPre = X(i,:)*Theta1(i,:)';
  %predictionsSigmoid = sigmoid(predictionsPre); %hypothesis of x
  %predictionsSigmoid1s = [ones(m, 1) predictionsSigmoid];
  %predictionsPre2 = PredictionsSigmoid1s *Theta2(i,:)';
  %predictionsSigmoid2 = sigmoid(predictionsPre2);
  %predictions(i)=predictionsSigmoid2;
%endfor
% building the Y matrix of results
I = eye(num_labels);
Y = zeros(m, num_labels);
for i = 1 : m
  Y(i, :) = I(y(i), :);
end

size(Theta1);
Z2 = X*Theta1';
A2 = sigmoid(X*Theta1');

A2 = [ones(m, 1) A2];
size(Theta2);
Z3 =A2*Theta2';
A3 = sigmoid(Z3);

%remove the first column used for bias
Jreg = lambda / (2 * m) * (sum(sum(Theta1(:, 2:end) .^2)) + sum(sum(Theta2(:, 2:end) .^2)));
%multiply -y times the log of each value in A3
%subtract from a
%sum accross column
%sum accross rows
%only sum for predictions that are the values we want.  thus use Y which is zero if not the value we need.
J = (1/m* sum(sum(((-Y .* log(A3)) - ((1-Y).*log(1-A3)))))) + Jreg;

d3 = A3 - Y;
d2 = (d3*Theta2 .*sigmoidGradient([ones(size(Z2),1) Z2]))(:,2:end);

% step 4
Delta2 = d3' * A2;
Delta1 = d2' * X;

% step 5
Theta1_grad = 1 / m * Delta1 + lambda / m * [zeros(size(Theta1), 1) Theta1(:, 2:end)];
Theta2_grad = 1 / m * Delta2 + lambda / m * [zeros(size(Theta2), 1) Theta2(:, 2:end)];

%Theta1_grad = zeros(size(Theta1));
%Theta2_grad = zeros(size(Theta2));

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



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
