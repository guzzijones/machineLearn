function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values

%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples
predictions = X*theta; % 1 + xi yi
differences= predictions - y;
diffSquared = differences.^2; %square each element

% You need to return the following variables correctly 
J = (1/(2*m))*sum(diffSquared);
% You need to return the following variables correctly 

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.





% =========================================================================

end