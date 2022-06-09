# -*- coding: utf-8 -*-
"""
Created on Tue May 17 14:04:02 2022

@author: Sai pranay
"""
#--------------------------importing the data set------------------------------

import pandas as pd
bk = pd.read_csv("E:\\DATA_SCIENCE_ASS\\ASSOCIATION RULES\\book.csv")
print(bk)
list(bk)
bk.describe()
bk.info()
bk.dtypes
bk.head()
bk.hist()
bk.shape


#---------------------checking_for_null_values---------------------------------

bk.isnull().sum()

#-----------------------GETTIG_DUMMY_VALUE-------------------------------------

m_m=pd.get_dummies(bk)
print(m_m)
m_m.shape
m_m.info()


from mlxtend.frequent_patterns import apriori,association_rules
from mlxtend.preprocessing import TransactionEncoder


#---------------------------Apriori Algorithm----------------------------------
product = apriori(m_m, min_support=0.1, use_colnames=True)
print(product)


rule = association_rules(product, metric="lift", min_threshold=0.7)
rule


rule.sort_values('lift',ascending = False)

rule.sort_values('lift',ascending = False)[0:20]

rule[rule.lift>1]

rule[['support','confidence']].hist()

rule[['support','confidence','lift']].hist()



import matplotlib.pyplot as plt

plt.scatter(rule['support'], rule['confidence'])
plt.show()


import seaborn as sns
sns.scatterplot('support', 'confidence', data=rule, hue='antecedents')

plt.show()
