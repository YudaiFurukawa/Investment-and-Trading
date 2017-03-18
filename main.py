import sklearn
import quandl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#regressor
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR #for regression
from sklearn import grid_search

#
from sklearn import tree
from sklearn.metrics import r2_score

# Quandl.api_config.api_key = "shPZyYs9kg4jQYxYZPbL"

#database list 
#WIKI
#YALE snp, fundamentals 
#BEA fundamentals
#ML latest bond yield

#database extra
# BOJ

###import data 
snp = quandl.get('YALE/SPCOMP') #SNP price, earnign, CPI, dividend
# yield1 = Quandl.get('ML/AAAEY') #AAA rate bond yield
#conf = Quandl.get('YALE/US_CONF_INDEX_VAL_INST') #institutional confidence level
# house = Quandl.get('YALE/RHPI') #housing price
# wage = Quandl.get('BEA/NIPA_2_7B_M') #wage

###reset index
snp = snp.reset_index(drop=True)

###drop columns not necessary
def delcol(data,columns):
	for key in columns:
		del data[key]
delcol(snp,['Real Price','Real Earnings','Real Dividend'])

###new arrays: monthly changes
monthly_changes = (snp/snp.shift(12))-1
monthly_changes = monthly_changes.fillna(value =0)
monthly_changes['y'] = monthly_changes['S&P Composite'].shift(-1) #snp['S&P Composite'].pct_change(periods = 12).dropna()#.reset_index(drop = True)


###new col
monthly_changes['PE Ratio'] = snp['Cyclically Adjusted PE Ratio']


###drop "S&P Composite"
delcol(monthly_changes,['CPI','Long Interest Rate'])

monthly_changes = monthly_changes[12:].dropna().reset_index(drop = True) #deleting first 12 rows as PE change doesn't have value

print(monthly_changes)

###plot
# monthly_changes.plot()
# plt.show()
###todo 
# make a new column with discrete target values


#print keys
# print("keys:", snp.keys())
# print(snp)


###display data stat
from IPython.display import display
# display(snp.describe())
display(monthly_changes.describe())

#scatter matrix
from pandas.tools.plotting import scatter_matrix
scatter_matrix(monthly_changes, alpha = 0.3, figsize = (14,8), diagonal = 'kde');

###correlation  matrix
print(pd.DataFrame(monthly_changes).corr())
###

###grid search, choose estimator
estimator = grid_search.GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=5, param_grid={"C": [1e0, 1e1, 1e2, 1e3],"gamma": np.logspace(-2, 2, 5)})

###r2 score
def r2(key,data, regressor):
	new_data = data.drop(key, axis = 1)
	target = data[key]
	X_train, X_test, y_train, y_test = train_test_split(new_data, target, test_size=0.25, random_state=7)
	regressor.fit(X_train.values,y_train.values)
	y_pred = regressor.predict(X_test.values)
	score = r2_score(y_test, y_pred)
	print("r2 score", score)
r2('y',monthly_changes,estimator)

# print(estimator.best_params_, estimator.best_estimator_)

#samples to see the result of prediction
print estimator.predict(monthly_changes.ix[1624,:].drop('y').values) #estimator.predict()


#plot
# plt.show()