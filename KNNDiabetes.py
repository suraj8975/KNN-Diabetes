
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

df = pd.read_csv("datasets/diabetes.csv")
df.head()
df.shape
df.columns
df.isna().sum()
X = df.drop(["Outcome"], axis=1)
y = df["Outcome"]
X.shape
y.shape
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
"accuracy score"
metrics.accuracy_score(y_test, y_pred)


from sklearn.metrics import confusion_matrix
"extracting true_positives, false_positives, true_negatives, false_negatives"
print(confusion_matrix(y_test, y_pred))
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print("True Negatives: ",tn)
print("False Positives: ",fp)
print("False Negatives: ",fn)
print("True Positives: ",tp)

"Accuracy"
Accuracy = (tn+tp)*100/(tp+tn+fp+fn)
print("Accuracy {:0.2f}%:".format(Accuracy))

"Precision"
Precision = tp/(tp+fp)
print("Precision {:0.2f}".format(Precision))

"Recall"
Recall = tp/(tp+fn)
print("Recall {:0.2f}".format(Recall))

"Error rate"
err = (fp + fn)/(tp + tn + fn + fp)
print("Error rate {:0.2f}".format(err))







