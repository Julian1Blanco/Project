

import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



from sklearn import datasets
from sklearn.cluster import KMeans
import sklearn.metrics as sm
import seaborn as sns



from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score
from sklearn import metrics


weld2 = pd.read_csv('weld2.csv')


weld2.head(5)



print(weld2.type.value_counts())


# Label encoding type column numerically
le = LabelEncoder()
le.fit(weld2['type'])
print(list(le.classes_))
iris['type'] = le.transform(weld2['type'])


print(iris['type'][0:5])
print(iris['type'][50:55])
print(iris['type'][100:105])


weld2.head(5)


matrix = pd.DataFrame.to_numpy(weld2[['rotation','temperature','time','shake']])
print(matrix)


cluster_model = KMeans(n_clusters=3, random_state=10)


print(cluster_model)


cluster_model.fit(matrix)


cluster_model.labels_


cluster_labels = cluster_model.fit_predict(weld2)
print(cluster_labels)


i=weld
print(i)


i.head(6)


i['pred'] = cluster_labels
print(i['pred'])


i.head(6)


sns.FacetGrid(i, hue="type", size=5).map(plt.scatter, "temperature", "rotation").add_legend()


sns.FacetGrid(i, hue="pred", height=5).map(plt.scatter, "temperature", "rotation").add_legend()


print(type(cluster_labels))


# Performance Metrics
sm.accuracy_score(weld2.type, cluster_model.labels_)


metrics.adjusted_rand_score(weld2.type, cluster_model.labels_)  #adjusted Rand index



# Confusion Matrix
sm.confusion_matrix(weld2.type, cluster_model.labels_)



