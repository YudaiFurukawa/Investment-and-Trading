import sklearn
import Quandl
import pandas as pd
import numpy as np

#database list 
#WIKI
#YALE snp, fundamentals 
#BEA fundamentals
#ML latest bond yield

#database extra
# BOJ

data_snp = Quandl.get('YALE/SPCOMP') #SNP price, earnign, CPI, dividend
data_yield = Quandl.get('ML/AAAEY') #AAA rate bond yield
data_conf = Quandl.get('YALE/US_CONF_INDEX_VAL_INST') #institutional confidence level
data_house = Quandl.get('YALE/RHPI') #housing price
data_wage = Quandl.get('BEA/NIPA_2_7B_M') #wage

df_snp = pd.DataFrame(data_snp)
df_yield = pd.DataFrame(data_yield)
df_conf = pd.DataFrame(data_conf)
df_house = pd.DataFrame(data_house)
df_wage = pd.DataFrame(data_wage)

#print keys
print(df_snp.keys(),df_yield.keys(),df_conf.keys(), df_house.keys(),df_wage.keys())

