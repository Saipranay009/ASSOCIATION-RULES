# -*- coding: utf-8 -*-
"""
Created on Mon May  9 17:39:40 2022

@author: Sai pranay
"""

import pandas as pd
mm = pd.read_csv("E:\DATA_SCIENCE_ASS\ASSOCIATION RULES\\my_movies.csv")
print(mm)
mm.shape
list(mm)
mm.describe()
mm.info()


mm1 = mm.drop(['V1','V2','V3','V4','V5'],axis = 1)
mm1

#---------------------checking_for_null_values---------------------------------

mm1.isnull().sum()

#-----------------------GETTIG_DUMMY_VALUE-------------------------------------

m_m=pd.get_dummies(mm1)
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







