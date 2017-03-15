import sklearn
import Quandl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split
from sklearn import tree
from sklearn.metrics import r2_score

#database list 
#WIKI
#YALE snp, fundamentals 
#BEA fundamentals
#ML latest bond yield

#database extra
# BOJ

###import data 
data_snp = Quandl.get('YALE/SPCOMP') #SNP price, earnign, CPI, dividend
# data_yield = Quandl.get('ML/AAAEY') #AAA rate bond yield
# data_conf = Quandl.get('YALE/US_CONF_INDEX_VAL_INST') #institutional confidence level
# data_house = Quandl.get('YALE/RHPI') #housing price
# data_wage = Quandl.get('BEA/NIPA_2_7B_M') #wage


###make dataframe
df_snp = pd.DataFrame(data_snp)
# df_yield = pd.DataFrame(data_yield)
# df_conf = pd.DataFrame(data_conf)
# df_house = pd.DataFrame(data_house)
# df_wage = pd.DataFrame(data_wage)

###reset index
df_snp = df_snp.reset_index(drop=True)

###drop columns not necessary
def delcol(df,columns):
	for key in columns:
		del df[key]
delcol(df_snp,['Real Price','Real Earnings','Real Dividend','Cyclically Adjusted PE Ratio'])

###add new columns
df_snp['pctChange'] = df_snp['S&P Composite'].pct_change(periods = 52).dropna().reset_index(drop = True)
df_snp = df_snp.dropna()

#print keys
print("keys:", df_snp.keys())
print(df_snp)
# df_snp.plot()
# plt.show()

###display data stat
from IPython.display import display
display(df_snp.describe())


###r2 score
#only for descrete velues
def r2(key):
	new_data = df_snp.drop(key, axis = 1)
	target = df_snp[key]#np.asarray(df_snp[key], dtype="|S6")#list(df_snp[key].values)
	X_train, X_test, y_train, y_test = train_test_split(new_data, target, test_size=0.25, random_state=7)
	regressor = tree.DecisionTreeClassifier(random_state=7)
	regressor.fit(X_train.values,y_train.values)
	y_pred = regressor.predict(X_test)
	score = r2_score(y_test, y_pred)
	print score
# r2('pctChange')
