import pandas as pd 
import numpy as np 
from math import log
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from matplotlib import pyplot

#log file contains the probability for positive class
data = pd.read_csv('/home/tharagesh/code_stuff/performance_metrics_code/log.csv')
log_loss = 0
eps = 1e-15
print(data.head())

true_positive = 40     
false_positive = 30    
true_negative = 20     
false_negative  = 10   

accuracy = float(true_positive+true_negative)/(true_positive+true_negative+false_positive+false_negative)
precision = float(true_positive)/(true_positive+false_positive)
recall = float(true_positive)/(true_positive+false_negative)
f1score = 2.0*(recall*precision)/(recall+precision)

errorrate = float(false_negative+false_positive)/(true_positive+true_negative+false_positive+false_negative)
far = float(false_positive)/(false_positive+true_negative)
frr = float(false_negative)/(false_negative+true_positive)

class_1_accuracy = float(true_positive)/(true_positive+false_negative)
class_2_accuracy = float(true_negative)/(false_positive+true_negative)
average_accuracy = float(class_1_accuracy+class_2_accuracy)/2.0

#log loss calculation
for index, row in data.iterrows():
    Y = float(row[0]) #actual value
    y = float(row[1]) #predicted value
    y = np.clip(y, eps, 1 - eps)
    log_loss = log_loss + (Y*(log(y)) + (1.0-Y)*(log(1.0-y)))

log_loss = -1*log_loss/data.shape[0] #log_loss/N

print("Accuracy        :" + str(accuracy))
print("Precision       :" + str(precision))
print("Recall          :" + str(recall))
print("Error rate      :" + str(errorrate))
print("F1 score        :" + str(f1score))
print("FAR             :" + str(far))
print("FRR             :" + str(frr))
print("Average accuracy:" + str(average_accuracy))
print("Log loss:" + str(log_loss))

#plot ROC curve
df = data.values
col1 = data.take([0], axis=1).values
col2 = data.take([1], axis=1).values
#print(df)
fpr, tpr, thresholds = roc_curve(col1, col2)
print("Thresholds for ROC curve")
print(thresholds) #print thresholds
pyplot.plot([0, 1], [0, 1], linestyle='--')
pyplot.plot(fpr, tpr, marker='.')
pyplot.show()

#plot precision-reacall curve
precision_PR, recall_PR, thresholds_PR = precision_recall_curve(col1, col2)
print("Thresholds for precision-recall curve")
print(thresholds_PR) #print thresholds
# plot no skill
pyplot.plot([0, 1], [0.5, 0.5], linestyle='--')
# plot the roc curve for the model
pyplot.plot(recall_PR, precision_PR, marker='.')
# show the plot
pyplot.show()

#DICE score calculation
DICE = (2.0*true_positive)/(2.0*true_positive+false_positive+false_negative)
print("DICE score     : " + str(DICE))

