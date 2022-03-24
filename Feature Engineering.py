#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import impyute


# In[2]:


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# In[3]:


data = pd.read_csv('Coronna_Data_CERTAIN_cleaned.csv')


# ### Processing 3MResponse

# In[4]:


data['3MResponse'].value_counts()


# In[5]:


#Delete the rows with a response of unknown, AS we do not want to the model to predict out 'unknown' result
data = data[data['3MResponse'] != 'Unknown']


# In[6]:


data['3MResponse'].value_counts()


# In[7]:


data


# In[8]:


#As we can see that for the IFN family, there are some rows that do not have values, and we want to fill in with zeros
data.info()


# ### Processing all IFN features, adding new flag feature

# In[9]:


#This is the function that flag whether IFN value is missing for a single row of data, and we will iterate over all the rows in IFN alpha and beta
#If IFNβ/α ratio final has value, it is 0, if not is nan, flag_na has a value of 1
def flag_na(row):
    if math.isnan(row['IFNβ/α ratio final']):
        return 1
    else:
        return 0


# In[10]:


#This line of code iterate over each row of IFN and add a new feature to indicate whether IFN is missing
data['IFN Missing'] = data.apply(lambda row: flag_na(row), axis = 1)


# In[11]:


data['IFN Missing'].value_counts()


# In[12]:


#After flagging, we can just fill in zero to make the variable a complete one
#here for IFN ratio, although zero will be meaningless for the ratio, we will use zero to identify the value-missing group
data = data.fillna({'Type I IFN activity':0, 'IFNβ activity final':0, 'IFNα activity final':0, 'IFNβ/α ratio final':0})


# In[13]:


data.columns


# In[14]:


#dropping 3M features because we want to predict based on BL values
data = data.drop(columns = ['seatedbp1', 'seatedbp2', 'pres_mtx', 'pres_arava', 'pres_azulfidine', 'pres_plaquenil',
                           'pres_pred', 'md_global_assess', 'pt_global_assess', 'di', 'pt_pain', 'usresultsIgA', 
                           'usresultsIgG', 'usresultsIgM'])


# In[15]:


data.shape


# ### Dropping features that will not be used in modelling

# In[16]:


#pres_imuran, pres_minocin, pres_minocin_BL, num_tnf, num_nontnf are all zeros
#This kind of variables will not provide useful information
data = data.drop(columns = ['pres_imuran', 'pres_minocin', 'pres_minocin_BL', 'num_tnf', 'num_nontnf'])


# In[17]:


#these are the columns that have correlation with the target feature
#We will need to delete these values to avoid redundent information and noise
data = data.drop(columns = ['ara_func_class', 'tender_jts_28', 'swollen_jts_28', 'usresultsCRP'])


# In[18]:


data.shape


# In[19]:


data.info()


# In[1]:


##### The TBD included in the title means that we have not decided on how to processing the certain features at that point
##### But later on we have processed them at the end of this file


# ### Processing ethnicity (TBD)

# In[20]:


#These are the rows that do not have ethnicity values, so I am wondering if I should get rid of these rows, or just fill in with zero
#This one I am thinking of using clustering to fill in the values (may introduce bais, but may also have some accuracy)
data[data['ethnicity'].isnull()]


# In[21]:


data['ethnicity']


# ### Processing final_education (TBD)

# In[22]:


data['final_education'].value_counts()


# In[23]:


#This person do not have final education, along with the ones who have don't remember values, I am also considering deleting them
data[data['final_education'].isnull()]


# ### Processing newsmoker (TBD)

# In[24]:


#A problem with all of these different columns is that they have null value in different rows, as a result, together they will reduce a large amount of data
#
data[data['newsmoker'].isnull()]


# In[25]:


#One thing I was thinking is using clustering to group the data, and impute the value based on similar values. 
#At the end, we used KNN, which is also using similarity, but is not a clustering model


# ### Processing drinker

# In[26]:


#These two lines I am just going to delete, as it has a lot of missing values in other columns as well
data[data['drinker'].isnull()]


# In[27]:


#Delete the rows with missing values
data = data[data['drinker'].notnull()]


# In[28]:


data.info()


# ### Processing useresultsRF

# In[29]:


#I will just delete it as it has a lot of missing values in this row
data[data['usresultsRF'].isnull()]


# In[30]:


#Because it does not have 0 value for normal data, thus here zero will indicate it has a missing value rather than a actual number
data = data[data['usresultsRF'].notnull()]


# In[31]:


data[data['usresultsCCP3'].isnull()]


# In[32]:


#For this, I just put zero to replace the nan values because the missing values are quite a lot, and we do not want to lose much information
data['usresultsCCP3']


# In[33]:


#Check if the feature has zero values itself. 
min(data['usresultsCCP3'])


# In[34]:


#Because it does not have 0 value for normal data, thus here zero will indicate it has a missing value rather than actual number
data['usresultsCCP3'] = data['usresultsCCP3'].replace(np.nan, 0)


# In[35]:


#For this two rows I checked in excel, it does have no value even in the BL sheet
data[data['seatedbp1_BL'].isnull()]


# In[36]:


min(data['seatedbp1_BL'])


# In[37]:


#Drops the rows with missing value for seatedbp1_BL
data = data[data['seatedbp1_BL'].notnull()]


# In[38]:


data[data['md_global_assess_BL'].isnull()]


# In[39]:


min(data['md_global_assess_BL'])


# In[40]:


data = data[data['md_global_assess_BL'].notnull()]


# In[41]:


data.info()


# In[42]:


#Not sure how to deal with these values
data[data['usresultsIgA_BL'].isnull()]


# In[43]:


#For the Ig values, i will also fill in with zeros as these are numerical values. 

data[['usresultsIgA_BL', 'usresultsIgG_BL', 'usresultsIgM_BL']]


# In[44]:


data = data.fillna({'usresultsIgA_BL': 0, 'usresultsIgG_BL':0, 'usresultsIgM_BL':0})


# In[45]:


#For this because the weight is unknown as well, I will just delete this row. 
data[data['BMI'].isnull()]


# In[46]:


#For weight, I am thinking about deleting it or using median to fill in? Another possibility is to use model to predict the weight value, but it is kind of unworth it.
data[data['weight'].isnull()]
#It will be somewhat strange to just add a zero here, we might just be deleting it


# In[47]:


data = data[data['BMI'].notnull()]


# In[48]:


data.info()


# In[49]:


data = data.drop(columns = ['UNMC_id'])


# ### From here, all the processing is after discussion, and will be final

# In[50]:


data[data['final_education'].isnull()]


# In[51]:


data['final_education'].value_counts()


# In[52]:


data[data['final_education'] == "don't remember"]


# In[53]:


plt.hist(data['usresultsRF'])


# In[54]:


data = data[data['final_education'].notnull()]
data = data[data['final_education'] != "don't remember"]


# In[55]:


data.info()


# ### Drop smkyrs and numcigs, which has too many missing values, which is not suitable to use in KNN Impute

# In[56]:


data = data.drop(columns = ['smkyrs', 'numcigs'])


# ### Processing ethnicity and newsmoker

# In[57]:


#We are using KNN imputation after consideration / I was going to use Hierarchical Cluster to mannualy do this, but I found a package that use KNN
#The idea is to use similarity to find similar data based on other features
from sklearn.impute import KNNImputer
from sklearn import preprocessing
from sklearn.preprocessing import scale


# In[58]:


#This cell is to find all the categorical variables, and assign each category an integer to make it into an ordinal variable
#This process is done for the scikit learn package, which only accepts all numerical inputs
encoders = dict()

for col_name in ['3MResponse', 'grp', 'gender', 'final_education', 'race_grp', 'newsmoker', 'drinker']:
    series = data[col_name]
    label_encoder = preprocessing.LabelEncoder()
    data[col_name] = pd.Series(
        label_encoder.fit_transform(series[series.notnull()]),
        index=series[series.notnull()].index
    )
    encoders[col_name] = label_encoder


# In[59]:


#Now all the features are numerical
data.info()


# ### Delete the mising rows in Eithnicity and Newsmoker (without KNN)

# In[60]:


data3 = data.copy()


# In[61]:


data3


# In[62]:


data3 = data3[data3['ethnicity'].notnull()]
data3 = data3[data3['newsmoker'].notnull()]


# In[63]:


#This table is used to test whether KNN helps us in prediction
data3.to_csv('Analytical_Base_Table_No_KNN.csv', index = False)


# ### Do not Delete, use KNN Imputation

# In[64]:


#This is a built in algorithm to built KNN model for Imputation
#n_neighbors controls the number of nearest points to be considered when calculating similarity
imputer = KNNImputer(n_neighbors=50, weights="uniform")


# In[65]:


data_temp = imputer.fit_transform(data)


# In[66]:


data


# In[67]:


data.info()


# In[68]:


data_temp = pd.DataFrame(data_temp, columns = data.columns)


# In[69]:


data_temp


# In[70]:


data


# In[71]:


data_temp.info()


# In[72]:


#This table is the major table we use in modelling
data_temp.to_csv('Analytical_Base_Table.csv', index = False)


# In[ ]:




