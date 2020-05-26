


import matplotlib
import matplotlib.pyplot as plt


import numpy as np
import pandas as pd
import statsmodels.api as sm


df = pd.read_csv("waste.csv")  




df.head(5)


df.shape


df.isnull().sum() 


df = df[["Survived","Pclass","Age","cost"]]


df=df.dropna() 


df.head(7)


plt.figure(figsize=(6,4))
fig, ax = plt.subplots()
df.Survived.value_counts().plot(kind='barh', color="blue", alpha=.65)
ax.set_ylim(-1, len(df.Survived.value_counts())) 
plt.title("Survival Breakdown (1 = Survived, 0 = Died)"):


sns.factorplot(x="Pclass", y="cost", hue="Survived", data=df, kind="box")



y=df[['Survived']]


print(type(y))


x=df[["Pclass","Age","cost"]]


print(type(x))


# Make model
logit =sm.Logit(y, x.astype(float))

# Fit model
result = logit.fit()


print result.summary()


# odds
print np.exp(result.params)


#prob = odds / (1 + odds) .



from patsy import dmatrices
import pandas as pd
from sklearn.linear_model import LogisticRegression
import statsmodels.discrete.discrete_model as sm


df2=pd.read_csv("waste.csv") 


df2.head(7)


df2 = df2[["Survived","Pclass","Sex","Age","cost"]]


df2.head(6)


df2=df2.dropna()


df2.head(6)


y, X = dmatrices('Survived ~ C(Pclass) + C(Sex) + Age + cost', df2, return_type = 'dataframe')


model = LogisticRegression(fit_intercept = False, C = 1e9)
mdl = model.fit(X, y)
model.coef_


logit = sm.Logit(y, X)
logit.fit().params



# Fit the model
result = logit.fit()
print result.summary()


# create a results dictionary
results = {} 

y,x = dmatrices(formula, data=df, return_type='dataframe')

# instantiate model
model = sm.Logit(y,x)

# fit model to the training data
res = model.fit()


results['Logit'] = [res, formula]
res.summary()


# cost is not statistically significant


formula = 'Survived ~ C(Pclass) + C(Sex) + Age' 



results = {} 

y,x = dmatrices(formula, data=df, return_type='dataframe')

# instantiate  model
model = sm.Logit(y,x)

# fit model to the training data
res = model.fit()

results['Logit'] = [res, formula]
res.summary()

