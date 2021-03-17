#import required libraries 
import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

data=pd.read_csv('INDIA.csv')
data.head(5)  #View first 5 row Eyeballing
"""
YEAR	INCOME
0	2011	1450
1	2012	1500
2	2013	1566
3	2014	1690
4	2015	1800
"""

data.describe()
"""

YEAR	INCOME
count	8.00000	8.000000
mean	2014.50000	1761.375000
std	2.44949	248.310829
min	2011.00000	1450.000000
25%	2012.75000	1549.500000
50%	2014.50000	1745.000000
75%	2016.25000	1987.000000
max	2018.00000	2099.000000

"""
#Visualication of data through scatter plot

%matplotlib inline
plt.scatter(data.YEAR,data.INCOME,COLOR='green',marker='+')
plt.xlabel("Years")
plt.ylabel("Income in us $ ")


reg = linear_model.LinearRegression()
reg.fit(data[['YEAR']],data.INCOME)
reg.predict([[2025]])  #model predicting the per capita income for the year 2025 

# array([2814.25]) India per capita for the year 2025 is 2025 us $  


