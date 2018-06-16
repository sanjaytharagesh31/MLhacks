fprintf("Visualisong data based on first two features\n");

data = load("data.txt");
X = data(:,1:4);
y = data(:,5);
m = size(X,1);

plotData(X, y);

fprintf('Program paused. Press enter to continue.\n');
pause;

fprintf('Training One-vs-All Logistic Regression...\n')

lambda = 0.1;
num_labels = 3;
[all_theta] = oneVsAll(X, y, num_labels, lambda);

fprintf('Program paused. Press enter to continue.\n');
pause;


pred = predict1vsAll(all_theta, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);
