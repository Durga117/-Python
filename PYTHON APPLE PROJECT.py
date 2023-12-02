#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="ticks")
import warnings
warnings.filterwarnings('ignore')


# In[2]:


Data = pd.read_csv('apple_products.csv')
Data


# In[3]:


Data.head()


# In[4]:


Data.shape


# In[5]:


Data.size


# In[6]:


Data.describe()


# ### Inference:
#     * In this we can see the summary statistics for the data, here the mean value for sale price is 80073.887097.
#     * we can see the mean, std,min, max,

# In[7]:


Data.describe(include='object')


# In[8]:


Data.info()


# ### Inference:
#     * There and 11 columns and 1 float type and 5 int and 5 object data types are available

# In[9]:


Data.isnull().sum()


# In[10]:


summary_stat = Data.describe().T

listing_columns = []

print(summary_stat)
print(listing_columns)


# In[11]:


x=Data.duplicated()
sum(x)


# In[12]:


Data.nunique()


# In[13]:


Data.drop('Product URL', axis=1, inplace=True)


# In[14]:


Data.drop('Upc', axis=1, inplace=True)


# In[51]:


Data.drop('Mrp', axis=1, inplace=True)


# In[52]:


Data.head()


# In[53]:


Data['Product Name'].unique()


# In[54]:


Data['Product Name'].value_counts()


# In[55]:


Data['Ram'].value_counts()


# In[56]:


Data['Star Rating'].value_counts()


# In[57]:


plt.figure(figsize=(10, 6))
plt.hist(Data['Sale Price'], bins=20, color='lightcoral', edgecolor='black')
plt.xlabel('Sale Price')
plt.ylabel('Frequency')
plt.title('Distribution of Sale Prices')
plt.show()


# ### Inference:
#     * This histogram displays the distribution of sale prices across the dataset.
#     * This histplot shows that maximum sales done in the price under 80,000 and in between 40000 and 60000

# In[58]:


plt.figure(figsize=(6, 5))
plt.scatter(Data['Sale Price'], Data['Star Rating'], color='green', alpha=0.5)
plt.xlabel('Sale Price')
plt.ylabel('Star Rating')
plt.title('Sale Price vs Star Rating')
plt.show()


# ### Inference:
#     * This scatter plot visualizes the relationship between sale price and star rating.
#     * We can see that maximum star rating gives for sale price under 40000 and 80000 is 4.6

# In[59]:


sns.pairplot(Data, hue = "Star Rating" )
plt.show()

* There is no correlation between anything 
* And we can also see that the maximum rating give is 4.6
# In[41]:


sns.boxplot(data=Data,x='Sale Price')
plt.title('Box plot for numerical variable')
plt.show()


# ### Inference:
#     * There is no outliers in this data
#     * Maximum people bought the Apple in the price under 60000 and 120000
#     * Sale price 80000 apple is maximum sold 

# In[61]:


corr_matrix = Data.corr()

corr_matrix


# In[60]:


plt.figure(figsize = (9,5))

sns.heatmap(Data.corr(),annot=True)
plt.show()


# ### Inference:
#     * Sale price has high correlation
#     * positive correlation:
#         * Discount percentage and Number of Ratings
#         * Discount percentage and Number of Reviews
#     * Negative correlation:
#         * Sale price and Star rating
#         * Star Rating and Number of Reviews
