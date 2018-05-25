function J = computeCost(X, y, theta)
	m = length(y);
	J = 0; %to return
	predict = X*theta;
	error = predict - y;
	J = (1/m) * (1/2) * sum(error.^2);
endfunction;