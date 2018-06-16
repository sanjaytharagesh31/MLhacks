function res = sigmoid(z)
	res = 1.0 ./ (1 + exp(-z));
end;