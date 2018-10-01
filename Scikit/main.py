from pandas import read_csv
from numpy import set_printoptions
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

filename = '/home/tharagesh/code_stuff/MOOC/scikit-learn/diabetes.csv'
dataframe = read_csv(filename)

data = dataframe.values
X = data[:, 0:8]
y = data[:, 8]

#print (y)

print(dataframe.groupby('Outcome').size())

test_size = 0.20
seed = 20

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state = seed)
model = RandomForestClassifier()
model.fit(X_train, y_train)
result = model.score(X_test, y_test)
print("Accuracy: " + str(result*100.0))