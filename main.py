import sklearn
import quandl
import pandas as pd
from pandas.tools.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display


#regressor
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR #for regression
from sklearn import grid_search, linear_model
from sklearn.neighbors import KNeighborsRegressor

#
from sklearn import tree
from sklearn.metrics import r2_score
from scipy import stats


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
# display(snp.describe())

#////////////////////////////////////////////
#############################################
###analysis 0: data exploration  
#############################################
# snp.plot()
# snp.plot.box()
# scatter_matrix(snp, alpha = 0.3, figsize = (14,8), diagonal = 'kde');
# plt.show()

# snp_timeSeries = snp.drop(["Real Price","S&P Composite"], axis = 1)
# snp_timeSeries.plot()
# plt.show()

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
snp_changes = (snp/snp.shift(12))-1
snp_changes = snp_changes.fillna(value =0)
snp_changes['y'] = snp_changes['S&P Composite'].shift(-1) #snp['S&P Composite'].pct_change(periods = 12).dropna()#.reset_index(drop = True)

###new col
snp_changes['PE Ratio (value)'] = snp['Cyclically Adjusted PE Ratio']
snp_changes['PER 6m change'] = (snp['Cyclically Adjusted PE Ratio']/snp['Cyclically Adjusted PE Ratio'].shift(6))-1
snp_changes['price 24m change']=(snp['S&P Composite']/snp['S&P Composite'].shift(24))-1
snp_changes['price 36m change']=(snp['S&P Composite']/snp['S&P Composite'].shift(36))-1
snp_changes['price 48m change']=(snp['S&P Composite']/snp['S&P Composite'].shift(48))-1
snp_changes['price 60m change']=(snp['S&P Composite']/snp['S&P Composite'].shift(60))-1

###drop "S&P Composite"
delcol(snp_changes,['CPI','Long Interest Rate'])

snp_changes = snp_changes[12:].dropna().reset_index(drop = True) #deleting first 12 rows as PE change doesn't have value

###p-value
x = snp_changes["S&P Composite"]
y = snp_changes["y"]
print stats.pearsonr(x, y)

###rename columns

#////////////////////////////////////////////

#############################################
###analysis 1: plotting 
#############################################

#print
# print(snp_changes)
# print snp['S&P Composite']


#outliers
for feature in snp_changes.keys():
	Q1 = np.percentile(snp_changes[feature],25)
	Q3  =np.percentile(snp_changes[feature],75)
	step = 1.5*(Q3 - Q1)
	print("Data points considered outliers for the feature '{}':".format(feature))
	display(snp_changes[~((snp_changes[feature] >= Q1 - step) & (snp_changes[feature] <= Q3 + step))])


###plot
snp_changes_timeSeries = snp_changes.drop(["PE Ratio (value)"], axis = 1)
snp_changes_timeSeries.plot()
# snp_changes["PE Ratio (value)"].plot()



##histogram
# snp_changes.hist(bins = 20)

# plt.axvline(Q1, color = 'b', linestyle ='dashed', linewidth =2)
# plt.axvline(Q3, color = 'b', linestyle ='dashed', linewidth =2)
# plt.axvline(Q1-step, color = 'r', linestyle ='dashed', linewidth =2)
# plt.axvline(Q3+step, color = 'r', linestyle ='dashed', linewidth =2)

# snp_changes.kurtosis()

##box plot
snp_changes_box = snp_changes.drop(["PE Ratio (value)"], axis = 1)
snp_changes_box.plot.box()
# snp_changes["PE Ratio (value)"].plot.box()

###todo 
# make a new column with discrete target values



###display data stat
display(snp.describe())
display(snp_changes.describe())

#scatter matrix
scatter_matrix(snp_changes, alpha = 0.3, figsize = (14,8), diagonal = 'kde');

###correlation  matrix
print(pd.DataFrame(snp_changes).corr())
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
	X_train, X_test, y_train, y_test = train_test_split(new_data, target, test_size=0.1, random_state=7)
	regressor.fit(X_train.values,y_train.values)
	y_pred = regressor.predict(X_test.values)
	score = r2_score(y_test, y_pred)
	print("r2 score", score)
r2('y',snp_changes,estimator)
# print(r2_score(snp_changes['y'][:1040].values,estimator.predict(snp_changes[:1040].drop('y').values)))

# print(estimator.best_params_, estimator.best_estimator_)
# print(estimator.alpha_)

#samples to see the result of prediction
# print estimator.predict(snp_changes.ix[1624,:].drop('y').values) 


#plot
plt.show()