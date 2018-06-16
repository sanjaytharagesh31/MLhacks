function plotData(X, y)

figure; hold on;

one = find(y==1);
two = find(y==2);
three = find(y==3);


plot(X(one, 1), X(one, 2), 'k+');
plot(X(two, 1), X(two,2), 'ko');
plot(X(three, 1), X(three, 2), 'kx');

hold off;

end;