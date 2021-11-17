
# coding: utf-8

# In[3]:

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import pandas as pd
from pandas.tools.plotting import scatter_matrix
import numpy as np
import seaborn as sns
import statistics as st
import re

df = pd.read_csv('IDS-Data-Collection.csv')

#Basic Operations on DataFrame
"""df.describe()
df.loc[2:4] 
df.head()
df.tail()
df.columns
df.shape"""

#Basic Querying using dataframe
#print(df[df['CGPA'] > 9]['CGPA'])
#print(df[(df.Sim_Course == "Yes") & (df.Focus == "Data")]['CGPA'])

#Cleaning CGPA 
for index, row in df.CGPA.iteritems():
    if row < 2.5 or row > 10:
        df.CGPA.loc[index] = np.median(df.CGPA)

#Cleaning Height
def isfloat(value):
    try:
        float(value)
        return True
    except:
        return False

for index, row in df['Height'].iteritems():
    if(isfloat(row)):
        if(float(row) < 10):
            df['Height'].loc[index] = float(row) * 30.48
        else :
            df['Height'].loc[index] = float(row)
    else :
        y = re.split("['|cm|mm]", row)
        if(isfloat(y[0])):
            if(float(y[0]) < 10):
                df['Height'].loc[index] =  float(y[0]) * 30.48 + float(y[1]) * 2.54
            else:
                df['Height'].loc[index] = float(y[0])
        else:
            df['Height'].loc[index] = np.NaN
        
df.Height = df.Height.fillna(np.median(df.Height))

for index, row in df['Height'].iteritems():
    if row <=100:
        df.Height.loc[index] = np.median(df.Height)
        
        
#Cleaning Weight Column
for index, row in df['Weight'].iteritems():
    if(isfloat(row)):
        df['Weight'].loc[index] = float(row)
    else :
        y = re.split("[kgs|Kg]", row)
        if(isfloat(y[0])):
            df['Weight'].loc[index] =  float(y[0])
        else:
            df['Weight'].loc[index] = np.NaN
            
df.Weight = df.Weight.fillna(np.mean(df.Weight))


#Cleaning sleep_duration
for index, row in df['sleep_duration'].iteritems():
    if(isfloat(row) and float(row) < 15):
        df['sleep_duration'].loc[index] = float(row)
    else:
        y = re.split("[-|:|hr|hrs|and]", row)
        if(isfloat(y[0]) and float(y[0]) < 15):
            df.sleep_duration.loc[index] = float(y[0])
        else:
            df.sleep_duration.loc[index] = np.NaN
            
df.sleep_duration = df.sleep_duration.fillna(np.mean(df.sleep_duration))

#Plotting Histograms using matplotlib; number of bins by default = 10 . (equal bin histogram)

freq, bins, patches = plt.hist(df.CGPA)  #fetching freq in each bin and bin values
plt.xlabel('CGPA')
plt.ylabel('Frequency')
plt.title('CGPA Histogram')
plt.show()
plt.hist(df.Height)
plt.xlabel('Height')
plt.ylabel('Frequency')
plt.title('Height Histogram')
plt.show()
plt.hist(df.Weight)
plt.xlabel('Weight')
plt.ylabel('Frequency')
plt.title('Weight Histogram')
plt.show()            
plt.hist(df.sleep_duration)
plt.xlabel('Sleep Duration ')
plt.ylabel('Frequency')
plt.title('Sleep Duration Histogram')
plt.show()

#plotting histogram using seaborn
sns.distplot(df.Height) #uses a standard rule to calculate number of bins and fits a plot on the histogram
plt.show()
#plotting histogram : df.Height.hist()

#plotting scatter plot
plt.scatter(df.Height, df.Weight)
plt.xlabel('Height ')
plt.ylabel('Weight')
plt.title('Height vs Weight')
plt.show()

plt.scatter(df.sleep_duration, df.Weight)
plt.xlabel('Sleep Duration ')
plt.ylabel('Weight')
plt.title('Sleep Duration vs. Weight')
plt.show()

plt.scatter(df.sleep_duration, df.CGPA)
plt.xlabel('Sleep Duration ')
plt.ylabel('CGPA')
plt.title('Sleep Duration vs. CGPA')
plt.show()



#Plotting histogram according to a categorical variable
grid = sns.FacetGrid(df, col='Gender')
grid.map(plt.hist, 'Height')
plt.show()

grid = sns.FacetGrid(df, col='Gender')
grid.map(plt.hist, 'CGPA')
plt.show()

grid = sns.FacetGrid(df, col='physical_activity')
grid.map(plt.hist, 'Weight')
plt.show()

grid = sns.FacetGrid(df, row='Focus', col='Sim_Course')
grid.map(plt.hist, 'CGPA')
plt.show()

grid = sns.FacetGrid(df, col='Rate_Programming')
grid.map(plt.hist, 'CGPA')
plt.show()

#Barcharts ; can also use plt.bar(no of categories, frequency)
df.Gender.value_counts().plot(kind="bar")
plt.xlabel('Gender')
plt.ylabel('Count')
plt.title('Bar Plot')
plt.show()
df.course_instructor.value_counts().plot(kind = "bar")
plt.xlabel('Course Instructor ')
plt.ylabel('Count')
plt.title('Bar Plot')
plt.show()
df.Rate_Programming.value_counts().plot(kind = "bar")
plt.xlabel('Programming')
plt.ylabel('Count')
plt.title('Bar Plot')
plt.show()
df.Focus.value_counts().plot(kind = "bar")
plt.xlabel('Focus : Data or Science')
plt.ylabel('Count')
plt.title('Bar Plot')
plt.show()
df.Engg_Choice.value_counts().plot(kind = "bar")
plt.xlabel('Engineering by Choice')
plt.ylabel('Count')
plt.title('Bar Plot')
plt.show()
df.physical_activity.value_counts().plot(kind="bar")
plt.xlabel('Physical Activity')
plt.ylabel('Count')
plt.title('Bar Plot')
plt.show()
df.prefer_maths.value_counts().plot(kind = "bar")
plt.xlabel('Prefer Maths')
plt.ylabel('Count')
plt.title('Bar Plot')
plt.show()


#Barplots : Plotting averages, confidence intervals must be specified
g = sns.FacetGrid(df)  
g.map(sns.barplot, 'Rate_Programming',"CGPA", ci = 95)  
plt.show()

#Grouped Bar Charts
sns.factorplot(x="Rate_Programming", y="CGPA", hue="Gender", data=df,kind="bar", ci = None)
plt.show()

x= sns.factorplot(x="physical_activity", y="Weight", hue="Gender", data=df,kind="bar", ci = None)
x.set_xticklabels(rotation=90)
plt.show()

#comparative box plots
sns.boxplot(x='Gender', y='CGPA', data=df)
plt.ylim(0, 11)
plt.show()

sns.boxplot(x='physical_activity', y='CGPA', data=df)
plt.ylim(0, 11)
plt.show()

#Swarmplots : Not a part of Syllabus but interesting to look at 
def focus_to_numeric(x):
    if x=='Data':
        return 1
    if x=='Science':
        return 0

def similar_course_to_numeric(x):
    if x=='Yes':
        return 1
    if x=='No':
        return 0
    
def sci_analytics_to_numeric(x):
    if x=='Yes':
        return 1
    if x=='No':
        return 0
    if x=='May be':
        return -1


df['focus_num'] = df['Focus'].apply(focus_to_numeric)
df['similar_course_num'] = df['Sim_Course'].apply(similar_course_to_numeric)
df['DS_DA_num'] = df['DS_DA'].apply(sci_analytics_to_numeric)
sns.swarmplot(x="similar_course_num", y="Focus", hue="Gender", data=df);
plt.show()
sns.swarmplot(x="DS_DA_num", y="similar_course_num", hue="Gender", data=df);
plt.show()


#Summary of the Datafram
pd.scatter_matrix(df, figsize=(20,20))
plt.show()

#Writing the changes made to the Dataframe to another file
df.to_csv('Cleaned.csv')


# In[ ]:



