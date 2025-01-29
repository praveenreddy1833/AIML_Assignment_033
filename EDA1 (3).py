#!/usr/bin/env python
# coding: utf-8

# In[1]:


#load the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data = pd.read_csv("data_clean.csv")
print(data)


# In[3]:


data.info()


# In[4]:


# DataFrame attributes
print(type(data))
print(data.shape)
print(data.size)


# In[5]:


# Drop duplicate column( Temp c)and Unnamed column
#data1 = data.drop(['Unnamed: 0',"Temp C"], axis =1, inplace = True)

data1 = data.drop(['Unnamed: 0',"Temp C"], axis =1)
data1


# In[6]:


# Convert the month column data type to float data type

data1['Month']=pd.to_numeric(data['Month'],errors='coerce')
data1.info()


# In[7]:


# Print all duplicated rows
data1[data1.duplicated(keep = False)]


# In[8]:


## Checking for duplicated rows in the table
# print only the duplicated row(one) only
data1[data1.duplicated()]


# In[9]:


# Drop duplicated rows
data1.drop_duplicates(keep='first', inplace = True)
data1


# Rename the columns

# In[10]:


# Change column names(Rename the columns)
data1.rename({'Solar.R': 'Solar'}, axis=1, inplace = True)
data1


# Impute the missing values

# In[11]:


data.info()


# In[12]:


# Display data1 missing values count in each column using isnull().sum()
data1.isnull().sum()


# In[13]:


# visualize data1 missing values using graph

cols = data1.columns
colors = ['black', 'white']
sns.heatmap(data1[cols].isnull(),cmap=sns.color_palette(colors),cbar = True)


# In[14]:


# Find the mean and median values of each numeric column
# Imputation of missing value with median
median_ozone = data1["Ozone"].median()
mean_ozone = data1["Ozone"].mean()
print("Meadian of Ozone: ", median_ozone)
print("Mean of Ozone: ", mean_ozone)


# In[15]:


# Replace the Ozone missing values with median value
data['Ozone'] = data1['Ozone'].fillna(median_ozone)
data1.isnull().sum()


# In[16]:


# print the data1 5 rows
data1.head()


# In[17]:


# Find the mode values of categorical column (weather)

print(data1["Weather"].value_counts())
mode_weather = data1["Weather"].mode()[0]
print(mode_weather)


# In[18]:


# Impute missing values (Replace NaN with mode etc.) using filna()
data1["Weather"] = data1["Weather"].fillna(mode_weather)
data1.isnull().sum()


# In[19]:


print(data1["Month"].value_counts())
mode_month = data1["Month"].mode()[0]
print(mode_month)


# In[20]:


data1["Month"] = data1["Month"].fillna(mode_month)
data1.isnull().sum()


# In[21]:


# reset the index column
data1.reset_index(drop=True)


# In[23]:


fig, axes = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={'height_ratios': [1, 3]})


sns.boxplot(data=data1["Ozone"], ax=axes[0], color='skyblue', width=0.5, orient='h')
axes[0].set_title("Boxplot")
axes[0].set_xlabel("Ozone levels")


sns.histplot(data1["Ozone"], kde=True, ax=axes[1], color='purple', bins=30)
axes[1].set_title("Histogram with KDE")
axes[1].set_xlabel("Ozone levels")
axes[1].set_ylabel("Frequency")


plt.tight_layout()
plt.show()


# In[26]:


fig, axes = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={'height_ratios': [1, 3]})

# Boxplot
sns.boxplot(data=data1["Solar"], ax=axes[0], color='skyblue', width=0.5, orient='h')
axes[0].set_title("Boxplot")
axes[0].set_xlabel("Solar levels")

# Histogram with KDE
sns.histplot(data1["Solar"], kde=True, ax=axes[1], color='purple', bins=30)
axes[1].set_title("Histogram with KDE")
axes[1].set_xlabel("Solar levels")
axes[1].set_ylabel("Frequency")

# Adjust layout and show
plt.tight_layout()
plt.show()


# In[30]:


plt.figure(figsize=(6,2))
boxplot_data=plt.boxplot(data1["Ozone"],vert=False)
[item.get_xdata() for item in boxplot_data['filters']]


# In[31]:


data1['Ozone'].describe()


# In[33]:


mu=data1["Ozone"].describe()[1]
sigma =data1["Ozone"].describe()[2]
for x in data1["Ozone"]:
    if ((x  <(mu - 3 *sigma)) or (x  >(mu  +3 *sigma))):
        print(x)


# In[34]:


import scipy.stats as stats
plt.figure(figsize=(8,6))
stats.probplot(data1["Ozone"],dist="norm",plot=plt)
plt.title("Q-Q plot for outlier detection",fontsize=14)
plt.xlabel("Theoretical quantiles",fontsize=12)


# In[35]:


import scipy.stats as stats
plt.figure(figsize=(4,6))
stats.probplot(data1["Solar"],dist="norm",plot=plt)
plt.title("Q-Q plot for outlier detection",fontsize=14)
plt.xlabel("Theoretical quantiles",fontsize=12)


# Observations from Q-Q plot
# The data does not follow normal distribution as the data points are deviating significantly away from the red line 
# The data shows a right skewed distribution and possible outliers

# In[36]:


sns.violinplot(data=data1, x = "Weather", y="Ozone", palete="Set2")


# In[43]:


sns.swarmplot(data=data1, x = "Weather", y = "Ozone",color="orange",palette="Set2", size=6)


# In[52]:


sns.stripplot(data=data1, x = "Weather", y = "Ozone",color="orange", palette="Set1", size=6,jitter=True)


# In[53]:


sns.kdeplot(data=data1["Ozone"], fill=True, color="blue")
sns.rugplot(data=data1["Ozone"], color="black")


# In[54]:


# Caterogy wise boxplot for ozone
sns.boxplot(data = data1, x = "Weather", y="Ozone")


# Correlation Coefficient and pair plots

# In[55]:


plt.scatter(data1["Wind"], data1["Temp"])


# In[56]:


# Compute pearson correlation coefficient
# between Wind spped and Temperature
data1["Wind"].corr(data1["Temp"])


# In[ ]:




