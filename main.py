import sklearn
import quandl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#regressor
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR #for regression
from sklearn import grid_search, linear_model
from sklearn.neighbors import KNeighborsRegressor

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

#############################################
###import data 
#############################################
snp = quandl.get('YALE/SPCOMP') #SNP price, earnign, CPI, dividend
# yield1 = quandl.get('ML/AAAEY') #AAA rate bond yield 1996-2017
# conf = quandl.get('YALE/US_CONF_INDEX_VAL_INST') #institutional confidence level every 6 months 1989-
# house = quandl.get('YALE/RHPI') #housing price annual 1890
# wage = quandl.get('BEA/NIPA_2_7B_M') #wage monghly 2001

# print(yield1, conf, house, wage)
# print(snp)

#////////////////////////////////////////////

#############################################
###modify datasets
#############################################
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
monthly_changes['PE Ratio (value)'] = snp['Cyclically Adjusted PE Ratio']
monthly_changes['PER 6m change'] = (snp['Cyclically Adjusted PE Ratio']/snp['Cyclically Adjusted PE Ratio'].shift(6))-1
monthly_changes['price 24m change']=(snp['S&P Composite']/snp['S&P Composite'].shift(24))-1
monthly_changes['price 36m change']=(snp['S&P Composite']/snp['S&P Composite'].shift(36))-1
monthly_changes['price 48m change']=(snp['S&P Composite']/snp['S&P Composite'].shift(48))-1
monthly_changes['price 60m change']=(snp['S&P Composite']/snp['S&P Composite'].shift(60))-1
###drop "S&P Composite"
delcol(monthly_changes,['CPI','Long Interest Rate'])

monthly_changes = monthly_changes[12:].dropna().reset_index(drop = True) #deleting first 12 rows as PE change doesn't have value

###rename columns

#////////////////////////////////////////////

#############################################
###analysis 1: plotting 
#############################################

#print
print(monthly_changes)
# print snp['S&P Composite']
###plot
# monthly_changes.plot()

#histogram
###have to figure out how to plot vlines on each chart
# Q1 = np.percentile(monthly_changes,25)
# Q3  =np.percentile(monthly_changes,75)
# step = 1.5*(Q3 - Q1)
# plt.axvline(Q1, color = 'b', linestyle ='dashed', linewidth =2)
# plt.axvline(Q3, color = 'b', linestyle ='dashed', linewidth =2)
# plt.axvline(Q1-step, color = 'r', linestyle ='dashed', linewidth =2)
# plt.axvline(Q3+step, color = 'r', linestyle ='dashed', linewidth =2)
monthly_changes.hist(bins = 20)
# monthly_changes.kurtosis()

#box plot
# monthly_changes.plot.box()

#plt.show()
###todo 
# make a new column with discrete target values



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
# estimator = grid_search.GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=5, param_grid={"C": [1e0, 1e1, 1e2, 1e3],"gamma": np.logspace(-2, 2, 5)})
# estimator = grid_search.GridSearchCV(KNeighborsRegressor(), param_grid={"n_neighbors": [2,3,4,5,6,7,8,9,10]})
estimator = linear_model.RidgeCV(alphas=[0.1, 0.5,1.0, 10.0])

#////////////////////////////////////////////


#############################################
###analysis 2: statistics 
#############################################

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
# print(r2_score(monthly_changes['y'][:1040].values,estimator.predict(monthly_changes[:1040].drop('y').values)))

# print(estimator.best_params_, estimator.best_estimator_)
print(estimator.alpha_)

#samples to see the result of prediction
# print estimator.predict(monthly_changes.ix[1624,:].drop('y').values) 


#plot
plt.show()