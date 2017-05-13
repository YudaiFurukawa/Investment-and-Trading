import sklearn
import quandl
import pandas as pd
from pandas.tools.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
from collections import Counter


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
###didn't improve the score
# snp_changes['price 36m change']=(snp['S&P Composite']/snp['S&P Composite'].shift(36))-1
# snp_changes['price 48m change']=(snp['S&P Composite']/snp['S&P Composite'].shift(48))-1
# snp_changes['price 60m change']=(snp['S&P Composite']/snp['S&P Composite'].shift(60))-1

###drop "S&P Composite"
delcol(snp_changes,['CPI','Long Interest Rate'])
snp_changes_testing = snp_changes[12:] #for testing in the end
snp_changes = snp_changes[12:].dropna(axis = 0).reset_index(drop = True) #deleting first 12 rows as PE change doesn't have value


###rename columns

#////////////////////////////////////////////

#############################################
###analysis 1: plotting 
#############################################

#print
# print(snp_changes)
# print(snp_changes.keys())


#outliers
outliersList = []
for feature in snp_changes.keys():
	Q1 = np.percentile(snp_changes[feature],25)
	Q3  =np.percentile(snp_changes[feature],75)
	step = 1.5*(Q3 - Q1)
	###print outliers for each feature
	# print("Data points considered outliers for the feature '{}':".format(feature))
	# display(snp_changes[~((snp_changes[feature] >= Q1 - step) & (snp_changes[feature] <= Q3 + step))])
	outliers = snp_changes[~((snp_changes[feature] >= Q1 - step) & (snp_changes[feature] <= Q3 + step))].index.tolist()
	# print(outliers)
	outliersList += outliers
# print('these are outliers',Counter(outliersList))

###print outliers
outliers_indices = [607, 609, 610, 611, 612, 605, 606, 608, 622, 623, 624,1541, 1542, 1543, 1544, 1545, 1546, 1547, 1548, 1549, 1550]
outliers_earning = [497, 498, 499, 1541, 1542, 1543, 1544, 1545, 1546, 1547, 1548, 1549, 1550, 1551]
# print('outliers for the earning feature',snp_changes.ix[outliers_earning])
print('earnings>2',snp_changes[snp_changes['Earnings']>2].index.tolist())
# print(snp_changes.ix[outliers_indices])

###drop outliers
good_data = snp_changes.drop(snp_changes.index[outliers_indices]).reset_index(drop = True)

#export as csv to examine
# np.savetxt("file_name.csv", good_data, delimiter=",", fmt='%s')
###plot
good_data_timeSeries = good_data.drop(["PE Ratio (value)"], axis = 1)
good_data_timeSeries.plot()
# snp_changes["PE Ratio (value)"].plot()



##histogram
# snp_changes.hist(bins = 20)

# plt.axvline(Q1, color = 'b', linestyle ='dashed', linewidth =2)
# plt.axvline(Q3, color = 'b', linestyle ='dashed', linewidth =2)
# plt.axvline(Q1-step, color = 'r', linestyle ='dashed', linewidth =2)
# plt.axvline(Q3+step, color = 'r', linestyle ='dashed', linewidth =2)

# snp_changes.kurtosis()

##box plot
snp_changes_box = good_data.drop(["PE Ratio (value)"], axis = 1)
snp_changes_box.plot.box()
# snp_changes["PE Ratio (value)"].plot.box()
	

###display data stat
# display(snp.describe())
display(good_data.describe())

#scatter matrix
scatter_matrix(good_data, alpha = 0.3, figsize = (14,8), diagonal = 'kde');

###correlation  matrix
print(pd.DataFrame(good_data).corr())
###

###grid search, choose estimator
##before refinement
# estimator = SVR()
# estimator = KNeighborsRegressor()
# estimator = LinearRegression()
#after refinement
# estimator = grid_search.GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=5, param_grid={"C": [1e0, 1e1, 1e2, 1e3],"gamma": np.logspace(-2, 2, 5)})
# estimator = grid_search.GridSearchCV(KNeighborsRegressor(), param_grid={"n_neighbors": [2,3,4,5,6,7,8,9,10]})
# estimator = linear_model.RidgeCV(alphas=[0.1, 0.5,1.0, 10.0])
estimator = grid_search.GridSearchCV(LinearRegression(), param_grid =  {'fit_intercept':[True,False], 'normalize':[True,False], 'copy_X':[True, False]})


#////////////////////////////////////////////


#############################################
###analysis 2: statistics 
#############################################

###r2 score
def r2(key,data, regressor):
	new_data = data.drop(key, axis = 1)
	target = data[key]
	X_train, X_test, y_train, y_test = train_test_split(new_data, target, test_size=0.2, random_state=7)
	regressor.fit(X_train.values,y_train.values)
	y_pred = regressor.predict(X_test.values)
	score = r2_score(y_test, y_pred)
	print("r2 score", score)
# r2('y',good_data,estimator)

def split(X, n_splits, groups=None):
    # X, y, groups = indexable(X, y, groups)
    n_samples = len(X)
    n_folds = n_splits + 1
    if n_folds > n_samples:
        raise ValueError(
            ("Cannot have number of folds ={0} greater"
             " than the number of samples: {1}.").format(n_folds,n_samples))
    indices = np.arange(n_samples)
    test_size = (n_samples // n_folds)
    test_starts = range(test_size + n_samples % n_folds,
                        n_samples, test_size)
    for test_start in test_starts:
        yield (indices[:test_start],
			indices[test_start:test_start + test_size])

new_data = good_data.drop('y', axis = 1)
target = good_data['y']
# print(good_data.ix[[605]])


for train_index, test_index in split(new_data, n_splits = 3):
	# print("TRAIN:", train_index, "TEST:", test_index)
	new_data_train, new_data_test = new_data.ix[train_index], new_data.ix[test_index]
	target_train, target_test = target.ix[train_index], target.ix[test_index]
	# print(list(map(tuple, np.where(np.isnan(new_data_train)))))
	# print(new_data_train.ix[[605]])
	###any nan or infinite
	# print(np.any(np.isnan(new_data_train)),np.all(np.isfinite(new_data_test)),
	###
	estimator.fit(new_data_train,target_train)
	# print(new_data_test)
	target_pred = estimator.predict(new_data_test.values)
	score = r2_score(target_test, target_pred)
	print("r2 score:",score)

# print(estimator.best_params_, estimator.best_estimator_)
# print(estimator.alpha_)
# print(estimator.best_estimator_.coef_, estimator.best_estimator_.residues_, estimator.best_estimator_.intercept_)

###samples to see the result of prediction
# print(good_data.ix[1624,'y'],estimator.predict(good_data.ix[1624,:].drop('y').values))
# print(good_data.ix[14,'y'],estimator.predict(good_data.ix[14,:].drop('y').values))
# print(good_data.ix[164,'y'],estimator.predict(good_data.ix[164,:].drop('y').values))
# print(good_data.ix[333,'y'],estimator.predict(good_data.ix[333,:].drop('y').values))
# print(good_data.ix[1000,'y'],estimator.predict(good_data.ix[1000,:].drop('y').values))
# print(estimator.predict(good_data.ix[333,:].drop('y').values))
# print(estimator.predict(snp_changes_testing.ix['2016-09-30',:].drop('y').values))
#plot
# plt.show()


### Time Series split





