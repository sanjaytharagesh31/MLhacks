from pandas import read_csv
from numpy import set_printoptions
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

#path to dataset
filename = 'diabetes.csv'
#read the csv file
dataframe = read_csv(filename)

#convert the data into array
data = dataframe.values
X = data[:, 0:8]
y = data[:, 8]

#print (y)

print(dataframe.groupby('Outcome').size())

#set test data sizee
test_size = 0.20
seed = 20

#split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state = seed)
#set model
model = RandomForestClassifier()
model.fit(X_train, y_train)
#validate the model
result = model.score(X_test, y_test)
print("Accuracy: " + str(result*100.0))
