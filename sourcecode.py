#data and plotting imports

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib

import matplotlib.pyplot as plt

#stats lib

from scipy.stats import skew
from scipy.stats.stats import pearsonr
from scipy import stats

#plotly imports

from plotly import tools
import plotly.plotly as py
import plotly.figure_factory as ff
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)


train = pd.read_csv('/Users/VyHo/Downloads/house price regression/train.csv')
test = pd.read_csv('/Users/VyHo/Downloads/house price regression/test.csv')

#keep the ids

train_id = train['Id']
test_id = test['Id']

train['SalePrice'].describe()

Missing = pd.concat([train.isnull().sum(),test.isnull().sum()], keys = ['train','test'], axis = 1)

#print rows with missing data only

Missing[Missing.sum(axis=1)>0]

#replace missing values with median or mean

train.info()

train.describe()

#heatmap

corr = train.corr()
plt.figure(figsize = (14,8))
plt.title('Overall Correlation of House Prices', fontsize = 18)
sns.heatmap(corr,annot=False, cmap = 'RdYlBu', linewidths = 0.2, annot_kws = {'size':20})
plt.show()

#create new dataframes with only desired columns

outsidesurr_df = train[['Id', 'MSZoning', 'LotFrontage', 'LotArea', 'Neighborhood', 'Condition1', 'Condition2', 'PavedDrive', 
                    'Street', 'Alley', 'LandContour', 'LandSlope', 'LotConfig', 'MoSold', 'YrSold', 'SaleType', 'LotShape', 
                     'SaleCondition', 'SalePrice']]

building_df = train[['Id', 'MSSubClass', 'BldgType', 'HouseStyle', 'YearBuilt', 'YearRemodAdd', 'RoofStyle', 
                    'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea', 'Foundation', 'Functional', 
                    'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'MoSold', 'YrSold', 'SaleType',
                    'SaleCondition', 'SalePrice']]

utilities_df = train[['Id', 'Utilities', 'Heating', 'CentralAir', 'Electrical', 'Fireplaces', 'PoolArea', 'MiscVal', 'MoSold',
                     'YrSold', 'SaleType', 'SaleCondition', 'SalePrice']]

ratings_df = train[['Id', 'OverallQual', 'OverallCond', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 
                   'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature',
                   'GarageCond', 'GarageQual', 'MoSold', 'YrSold', 'SaleType', 'SaleCondition', 'SalePrice']]

rooms_df = train[['Id', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BsmtFinSF1', 'BsmtFinSF2',
                 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF','TotRmsAbvGrd', 
                 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'MoSold', 'YrSold', 'SaleType',
                 'SaleCondition', 'SalePrice']]

# set Id as index of the dataframe

outsidesurr_df = outsidesurr_df.set_index('Id')
building_df = building_df.set_index('Id')
utilities_df = utilities_df.set_index('Id')
ratings_df = ratings_df.set_index('Id')
rooms_df = rooms_df.set_index('Id')

# move SalePrice to the first column (Our Label)

sp0 = outsidesurr_df['SalePrice']
outsidesurr_df.drop(labels = ['SalePrice'],axis = 1, inplace = True)
outsidesurr_df.insert(0,'SalePrice',sp0)

sp1 = building_df['SalePrice']
building_df.drop(labels = ['SalePrice'], axis = 1, inplace = True)
building_df.insert(0, 'SalePrice', sp1)

sp2 = utilities_df['SalePrice']
utilities_df.drop(labels = ['SalePrice'], axis = 1, inplace = True)
utilities_df.insert(0, 'SalePrice', sp2)


sp3 = ratings_df['SalePrice']
ratings_df.drop(labels = ['SalePrice'], axis = 1, inplace = True)
ratings_df.insert(0, 'SalePrice', sp3)

sp4 = rooms_df['SalePrice']
rooms_df.drop(labels=['SalePrice'], axis=1, inplace=True)
rooms_df.insert(0, 'SalePrice', sp4)

#histograms

import seaborn as sns
sns.set_style('white')

f,axes = plt.subplots(ncols=4, figsize= (16,4))

#lot area in square feet

sns.distplot(train['LotArea'], kde = False, color = "#DF3A01", ax = axes[0]).set_title("Distribution of LotAre")
             
axes[0].set_ylabel("Square Ft")
axes[0].set_xlabel("Amount of Houses")

# MoSold: Year of the Month sold
sns.distplot(train['MoSold'], kde=False, color="#045FB4", ax=axes[1]).set_title("Monthly Sales Distribution")
axes[1].set_ylabel("Amount of Houses Sold")
axes[1].set_xlabel("Month of the Year")

# House Value
sns.distplot(train['SalePrice'], kde=False, color="#088A4B", ax=axes[2]).set_title("Monthly Sales Distribution")
axes[2].set_ylabel("Number of Houses ")
axes[2].set_xlabel("Price of the House")

# YrSold: Year the house was sold.
sns.distplot(train['YrSold'], kde=False, color="#FE2E64", ax=axes[3]).set_title("Year Sold")
axes[3].set_ylabel("Number of Houses ")
axes[3].set_xlabel("Year Sold")

#right skew dist. mean is right of the median
plt.show()
            
plt.figure(figsize = (12,8))

sns.distplot(train['SalePrice'],color='r')
plt.title("Distribution of Sale Price",fontsize = 18)

plt.show()

#people tend to move during the summer
#horizontal histogram
sns.set(style='whitegrid')
plt.figure(figsize = (12,8))
sns.countplot(y='MoSold',hue='YrSold',data=train)
plt.show()

plt.figure(figsize = (12,8))

sns.boxplot(x='YrSold', y = 'SalePrice', data = train)
plt.xlabel('Year Sold', fontsize = 14)
plt.ylabel('Price Sold', fontsize = 14)
plt.title('House Sold per Year', fontsize = 16)

plt.figure(figsize = (14,8))
plt.style.use('seaborn-white')
sns.stripplot(x='YrSold', y = 'YearBuilt', data = train, jitter = True, palette = 'Set2', linewidth = 1)
plt.xlabel('Year the house was sold', fontsize = 18)
plt.ylabel('Year the house was built', rotation = 90,fontsize = 14)
plt.title('Economic Activity Analysis', fontsize = 18)

outsidesurr_df.describe()

outsidesurr_df.columns

#outside surroundings correlation to house price

plt.style.use('seaborn-white')
corr = outsidesurr_df.corr()

sns.heatmap(corr,annot=True,cmap = 'YlOrRd',linewidths= 0.2,annot_kws= {'size':20})
fig = plt.gcf()
fig.set_size_inches(14,10)
plt.title('Ouside Surroundings Correlation', fontsize = 18)
plt.xticks(fontsize = 14)
plt.yticks(fontsize=14)
plt.show()

#corr graph created for all numeric columns
#year sold doesnt affect price much
#lotfrontage = width of a lot

#which neighborhood gives the most revenuw

plt.style.use('seaborn-white')
zoning_value = train.groupby(by=['MSZoning'],as_index = False)['SalePrice'].sum()
zoning = zoning_value['MSZoning'].values.tolist()

#create a pie chart

labels = ['C: Commercial', 'FV: Floating Village Res.', 'RH: Res. High Density', 'RL: Res. Low Density', 'RM: Res. Medium Density']

total_sales = zoning_value['SalePrice'].values.tolist()
explode = (0,0,0,0.1,0)

fig, ax1 = plt.subplots(figsize = (12,8))

texts = ax1.pie(total_sales, explode=explode, autopct = '%.1f%%', shadow = True, startangle = 90,pctdistance = 0.8,radius = 0.5 )

ax1.axis('equal')
plt.title('Sales Groupby Zones', fontsize = 16)
plt.tight_layout()
plt.legend(labels,loc = 'best')
plt.show()

plt.style.use('seaborn-white')
SalesbyZone = train.groupby(['YrSold','MSZoning']).SalePrice.count()
SalesbyZone.unstack().plot(kind= 'bar', stacked = True, colormap = 'gnuplot', grid= False, figsize = (12,8))

plt.title('Building Sale (2006 - 2010) by Zoning', fontsize = 18)
plt.ylabel('Sale Price', fontsize = 14)
plt.xlabel('Sales per Year', fontsize = 14)

plt.show()

fig, ax = plt.subplots(figsize = (12,8))
sns.countplot(x = 'Neighborhood', data = train, palette = 'Set2')
ax.set_title('Types of Neighborhoods', fontsize = 20)
ax.set_xlabel('Neighborhoods',fontsize = 16)
ax.set_ylabel('Number of Houses Sold', fontsize = 16)
ax.set_xticklabels(labels=train['Neighborhood'],rotation = 90)
plt.show()

#sawyer and sawyerw tend to be the most expensive neighborhoods

fig, ax = plt.subplots(figsize = (12,8))
ax = sns.boxplot(x='Neighborhood',y='SalePrice', data = train)
ax.set_title('Range Value of the Neighborhoods', fontsize = 18)
ax.set_ylabel('Price Sold', fontsize = 16)
ax.set_xlabel('Neighborhood',fontsize = 16)
ax.set_xticklabels(labels=train['Neighborhood'],rotation = 90)
plt.show()

#building characteristics vs. saleprices

 
