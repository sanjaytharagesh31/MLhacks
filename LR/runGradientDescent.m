function [theta, costHistory] = runGradientDescent(iterations, alpha, theta, X, y)
	m = length(y);
	costHistory = zeros(iterations,1);
	for i = iterations,
		predict = X*theta;
		theta(1) = theta(1) - alpha * (1/m) * sum(predict-y);
		theta(2) = theta(2) - alpha * (1/m) * sum((predict-y) .* X(:,2));
		costHistory(i) = computeCost(X, y, theta);
	endfor;
endfunction;