

import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split


from sklearn.ensemble import RandomForestClassifier


glass=pd.read_csv("glassClass.csv")


glass.head(7)


X= glass.drop("Type", axis=1) #predictors
Y = glass["Type"] #predictor


# ## training & testing data

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = .2, random_state=25) #20% hold out for testing


random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)


random_forest.score(X_train, y_train) 


Y_pred = random_forest.predict(X_test) 
#predict the classification of glass based on test predictors


Y_pred #predicted values of classifications


random_forest.predict_proba(X_test)[0:10] #How confident is the classifier about each glass type? 


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, Y_pred)
confusion_matrix


from sklearn.metrics import accuracy_score


accuracy_score(y_test, Y_pred) #compare with the actual y values, y_test (hold outs) with predicted y


from sklearn.metrics import cohen_kappa_score
cohen_kappa_score(y_test, Y_pred) #Scores above .8 are generally considered good agreement


from sklearn.metrics import classification_report


report = classification_report(y_test, Y_pred)
print(report)

# ## feature selection


from sklearn.feature_selection import RFE  ## Recursive Feature Elimination (RFE) method 


# RFE model
rfe = RFE(random_forest, 5)
fit = rfe.fit(X, Y)

# summarize the selection of the attributes
print("Num Features: %d") % fit.n_features_
print("Selected Features: %s") % fit.support_
print("Feature Ranking: %s") % fit.ranking_

from sklearn.ensemble import ExtraTreesClassifier


forest = ExtraTreesClassifier(n_estimators=250,
                              random_state=0)

forest.fit(X, Y)
importances = forest.feature_importances_


indices = np.argsort(importances)[::-1]


print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

from sklearn.model_selection import KFold
from sklearn import model_selection
seed = 10
kfold = model_selection.KFold(n_splits=10, random_state=seed)





from sklearn.model_selection import cross_val_score
scores = cross_val_score(random_forest, X, Y, cv=5)


print(scores)





