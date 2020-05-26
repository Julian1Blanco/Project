
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns
import statsmodels.api as sm
from sklearn import linear_model



from scipy.stats import ttest_1samp, wilcoxon, ttest_ind, mannwhitneyu



import scipy.stats as stats


weld4 = pd.read_csv('weld4.csv')
weld4.head()



t_statistic, p_value = ttest_1samp(weld4["temperature"], 17)



print "one-sample t-test", p_value


import scipy


shapiro_results = scipy.stats.shapiro(weld4["temperature"])



print(shapiro_results)

z_statistic, p_value = wilcoxon(weld4["temperature"] - 17)
print "one-sample wilcoxon-test", p_value


t_statistic, p_value = ttest_ind(weld4["temperature"], weld4["rotation"])



print "two-sample t-test", p_value

u, p_value = mannwhitneyu(weld4["temperature"], weld4["rotation"])
print "two-sample wilcoxon-test", p_value



t_statistic, p_value = ttest_1samp(weld4["temperature"] - weld4["rotation"], 0)


print "paired t-test", p_value


z_statistic, p_value = wilcoxon(weld4["temperature"] - weld4["rotation"])

print "paired wilcoxon-test", p_value





