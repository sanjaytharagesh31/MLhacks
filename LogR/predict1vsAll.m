function[pre] = predict1vsAll(all_theta, X)
	m = size(X,1);
	X = [ones(m,1),X];
	num_labels = size(all_theta, 1);
	pre = zeros(size(X, 1), 1);
	[val, pre] = max(X*all_theta',[],2);
end;