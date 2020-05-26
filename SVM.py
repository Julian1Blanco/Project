
import numpy as np
import pandas as pd

from sklearn import svm, datasets
from sklearn import datasets 
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.svm import LinearSVC 

weld2 = pd.read_csv('weld2.csv')
X = weld2.data[:, :2]  
                      
Y = weld2.target

h = .02 


C = 1.0  # SVM regularization 


#split the data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = .25, random_state=25) #25% hold out for testing



svc = svm.SVC(kernel='linear', C=C) #initialize an SVM model



# fit the linear svm


svc.fit(X_train, y_train)



Y_pred = svc.predict(X_test)



#test accuracy


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, Y_pred)
confusion_matrix


from sklearn.metrics import accuracy_score


print accuracy_score(y_test, Y_pred)


