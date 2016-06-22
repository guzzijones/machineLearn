function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J=0;
grad = zeros(size(theta));
size(grad);
m = length(y); % number of training examples
predictions = X*theta;
predictionsSigmoid = sigmoid(predictions);

matrixTimesXPrime =  X'*(predictionsSigmoid-y);
multiplier = ones(size(theta));
multiplier=multiplier.*(lambda/(m));
multiplier(1,1)=0;
grad = 1/(m)* matrixTimesXPrime + (multiplier.*theta);


reg = multiplier' *theta;
thetawithoutfirst = theta([1])=[]
%cost
J = (1/m* ((-y' * log(predictionsSigmoid)) - ((1-y)'*log(1-predictionsSigmoid)))) + (lambda / (2*m))*(theta'*theta);
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta






% =============================================================

end
