function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
%theta
%X
%y
J=0;
grad = zeros(size(theta));
size(grad);
m = length(y); % number of training examples
predictions = X*theta;
predictionsSigmoid = sigmoid(predictions);
%X'
matrixTimesXPrime =  X'*(predictionsSigmoid-y);
grad = (1/m)* matrixTimesXPrime;
%size(grad);
%cost
J = 1/m* ((-y' * log(predictionsSigmoid)) - ((1-y)'*log(1-predictionsSigmoid)));
% You need to return the following variables correctly 



% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%








% =============================================================

end
