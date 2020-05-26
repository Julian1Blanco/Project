

import matplotlib
import matplotlib.pyplot as plt


import numpy as np
import pandas as pd
import statsmodels.api as sm


import seaborn as sns

hosp = pd.read_csv('hosp.csv')
hosp.head()


x=hosp.duration
y=hosp.repetitions


from sklearn import linear_model


lr= linear_model.LinearRegression()


from sklearn.metrics import r2_score


for deg in [1,2,3,4,5]:
    lr.fit(np.vander(x,deg+1),y);
    y_lr=lr.predict(np.vander(x,deg+1))
    plt.plot(x,y_lr,label='degree'+str(deg));
    plt.legend(loc=2);
    print r2_score(y,y_lr)
    
plt.plot(x,y,'ok')    




