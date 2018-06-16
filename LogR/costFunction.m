function [J, grad] = costFunction(theta, X, y, lambda)
	m = size(y,1);
	J = 0;
	grad = zeros(size(theta));
	
	h = sigmoid(X*theta);
	
	J = (1/m) * sum((-y).*log(h)-(1-y).*log(1-h)) + ((lambda/(2*m)) * sum(theta(2:size(theta)).^2));
	grad(1) = (1/m) * sum(h-y);
	grad(2:size(theta)) = (1/m) *( X(:,[2:size(theta)])' * (h-y)) + (lambda/m)*theta(2:size(theta));
end;	