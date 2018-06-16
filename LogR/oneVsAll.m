function [all_theta] = oneVsAll(X, y, labels, lambda)

m = size(X, 1);
n = size(X, 2);

all_theta = zeros(labels, n+1);
X = [ones(m,1),X];


for i=1:labels,
	initial_theta = zeros(n+1, 1);
	options = optimset('GradObj', 'on', 'MaxIter', 100);
	[theta] = fmincg(@(t)(costFunction(t, X, (y==i), lambda)),initial_theta, options);
	all_theta(i,:) = theta;
endfor;