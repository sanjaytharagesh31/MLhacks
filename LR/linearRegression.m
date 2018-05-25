fprintf("Linear regreession sample\n");

%%ploting data

fprintf("Ploting Data\n");
data = load("ex1data1.txt");
X = data(:,1);
y = data(:,2);
m = length(y);
plotData1(X, y);
fprintf("Press Enter\n");
pause;

%%computing Cost

X = [ones(m,1), data(:,1)];
y = data(:,2);
theta = zeros(2,1);
J = computeCost(X, y, theta);
fprintf("Cost for theta1=0 theta2=0: %f\n", J);
fprintf("Press Enter\n");
pause;

%%Gradient Descent Algorithm

iterations = [1:2000];
alpha = 0.001;
[theta, costHistory] = runGradientDescent(iterations, alpha, theta, X, y);
fprintf("Value of theta after running gradient descent\n");
fprintf("%f\n", theta);
fprintf("Press Enter\n");
pause;

%%ploting hypothesis, linear linear
hold on;
h = theta(1) + theta(2) .* X(:,2);
plot(X(:,2), h, '-');
fprintf("Press Enter\n");
hold off;
pause;

%%ploting cost history
fprintf("ploting cost history\n");
plot(iterations, costHistory, '-');
fprintf("Press Enter\n");
pause;













