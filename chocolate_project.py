#By the end of this mini project, you will need to deliver within your code:
#The count of the tuples in the given dataset.
#The count of the names of unique company names from the attributes.
#The count of reviews in 2013 from the attributes.
#The count of missing values in a specific given.
#An output plot of the histogram of the values in the column named Ratings.
#An output plot of the scatter plot between the cocoa percent values against the rating values.
#The normalized ratings column values.
#You are expected to write around 25 lines of code to complete this project.


#Download the Dataset
#Download the dataset from the following link:
#https://www.kaggle.com/rtatman/chocolate-bar-ratings
#Download the dataset to your local computer in the project directory of your choice.
#Reading the Dataset
#Read the dataset into a Pandas DataFrame!
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


df = pd.read_csv('flavors_of_cacao.csv')
column_names = ['Company', 'SpecificBeanOriginorBarName', 'REF',
       'ReviewDate', 'CocoanPercent', 'CompanyLocation', 'Rating', 'BeanType',
       'BroadBeanOrigin']
df.columns = column_names



#In the BeanType Column, how many missing values are there?
#Does the dataset include any missing values? If so, delete the missing value entries!
#Hint: Pandas can do that with one line of code!
df = df.replace(r'^\s*$', np.nan, regex=True)
df = df.dropna()


#Exploring the Dataset
#Answer the following questions about the dataset using Python commands:
#How many tuples are there in the dataset?
numTuples = len(df)
print("num tuples:" , numTuples)



#How many unique company names are there in the dataset?
unique_companyNames = df['Company'].unique()
print("unique company names:",len(unique_companyNames))
unique_companyLocation = df['CompanyLocation'].unique()
print("unique_companyLocation:",len(unique_companyLocation))


#How many reviews are made in 2013 in the dataset?
num_reviews = sum(df["ReviewDate"]==2013)
print("num reviews:",num_reviews)




#Hint: Each question should require few lines of code!
#Visualization
#Visualize the rating column with a histogram!


plt.figure(figsize=(12,4))

plt.hist(df['Rating'], label = ['Rating'], bins = 50)

plt.title("Rating comparison", size = 15)
plt.xlabel("Rating")
plt.ylabel("Count")
plt.legend(loc='upper left', frameon=False)


#Comment on the resulting figure!


#Convert and Visualize.
#Convert the Column Percent.
#Change the type of values in the column percent from string values to numerical values.

df['CocoanPercent'] = df['CocoanPercent'].str.strip("%")


#Visualize
#Plot the converted numerical Cocoa Percent values against the Rating values!
#From what you see, does more cocoa in a bar correspond to a higher rating?
#Hint: Try a scatter plot with small alpha, e.g., 0.1, to flush out the density of each point.

plt.scatter(df["CocoanPercent"],df["Rating"],alpha = 0.1)


#Normalization
#Normalize the Rating Column and print the results.

from sklearn.preprocessing import StandardScaler
df['Rating']= StandardScaler().fit_transform(df['Rating'].values.reshape(-1,1))

#Challenge yourself (Optional)
#List the companies ordered by their average score (averaged over each company’s reviews).

Company_means = {}
for i in range(len(unique_companyNames)):
    Company_means[unique_companyNames[i]] = np.mean(df['Company'] == unique_companyNames[i], axis = 0)
    
    
sorted_Company_means = dict(sorted(Company_means.items(), key=lambda item: item[1]))

    
print("Companies ordered by their average score:  ",sorted_Company_means.keys())

#Encoding          
#Suppose we are interested in the company’s names and locations for some collective analysis. 
#Encode the two categorical columns with the encoder you think is best for the job!

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

df['Company'] = LabelEncoder().fit_transform(df['Company'].values.reshape(-1, 1))
df['CompanyLocation'] = LabelEncoder().fit_transform(df['CompanyLocation'].values.reshape(-1,1))
df
   
    