function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

%actual cost
Htheta = X*theta;
error = Htheta - y;
sqrerror = error.^2;
nonreg = sum(sqrerror);

%regularized portion
n = length(theta);
only = theta(2:n);
reg = (lambda)*(only' * only);

%total cost
J = (1/(2*m)) * (nonreg + reg);





% =========================================================================

%first gradient
a = X(1:m,1);
first = sum(error.*a)/m;

%second gradient
b = X(1:m,2);
second = sum(error.*b)/m + (lambda/m)*(theta(2));

grad(1) = first;
grad(2) = second;

end
