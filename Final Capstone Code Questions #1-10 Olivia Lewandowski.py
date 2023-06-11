#!/usr/bin/env python
# coding: utf-8

#QUESTION #1

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.stats import ranksums
import seaborn as sns
from scipy.stats import shapiro
from scipy.stats import ttest_rel
from scipy.stats import levene
from scipy.stats import wilcoxon

#import the dataset
art_data = pd.read_csv('theData.csv')
art_data.head()

#columns 1-35 represent classical art 
classical_art = art_data.iloc[:, :35]
#columns 36-70 represent modern art
modern_art = art_data.iloc[:, 35:70]

classical_art_list = classical_art.stack()
modern_art_list = modern_art.stack()
print(classical_art_list)

plt.hist(classical_art_list, bins=15)
plt.show()

plt.hist(modern_art_list, bins = 15)
plt.show()

#kde and ecdf plots
fig, ax = plt.subplots(ncols=2, figsize=(12,6))
sns.kdeplot(classical_art_list, ax=ax[0], shade=True, label='Classical Art Ratings')
sns.kdeplot(modern_art_list, ax=ax[0], shade=True, label='Energy Ratings')
ax[0].set_title('KDE plot')
ax[0].legend()

sns.ecdfplot(classical_art_list, ax=ax[1], label='Classical Art Ratings')
sns.ecdfplot(modern_art_list, ax=ax[1], label='Energy Ratings')
ax[1].set_title('ECDF plot')
ax[1].legend()

plt.show()

#comparative histogram
sns.histplot(data=classical_art_list, color='blue', alpha=0.5, label='Classical Pref. Ratings', bins=7, binrange=(0.5, 7.5))
sns.histplot(data=modern_art_list, color='red', alpha=0.5, label='Modern Pref. Ratings', bins=7, binrange=(0.5, 7.5))

#labels and legend
plt.xlabel('Art Preference Ratings')
plt.ylabel('Frequency')
plt.xticks(np.arange(1, 8))
plt.legend(loc='upper left')

plt.ylim(0, max(max(np.histogram(classical_art_list, bins=7)[0]), max(np.histogram(modern_art_list, bins=7)[0])) + 10)


#plot
plt.show()

classical_column_means = classical_art.mean(axis=0)
classical_mean = classical_column_means.mean()
print(classical_mean)

classical_std = classical_art.stack().std()
print(classical_std)

modern_column_means = modern_art.mean(axis=0)
modern_mean = modern_column_means.mean()
print(modern_mean)

modern_std = modern_art.stack().std()
print(modern_std)

#test for normality - classical
stat, p = shapiro(classical_art_list)

#results
alpha = 0.05
if p > alpha:
    print('normal')
else:
    print('non-normal')

print("p-value: {:.4f}".format(p))

#test for normality - modern
stat, p = shapiro(modern_art_list)

#results
alpha = 0.05
if p > alpha:
    print('normal')
else:
    print('non-normal')
    
print("p-value: {:.4f}".format(p))

# check for equal variances
stat, p = levene(classical_art_list, modern_art_list)

if p > 0.05:
    print('variances are equal')
else:
    print('variances are not equal')

classical_median = np.median(classical_art_list)
modern_median = np.median(modern_art_list)

print("Median of classical_art_list:", classical_median)
print("Median of modern_art_list:", modern_median)

stat, p_value = wilcoxon(classical_art_list, modern_art_list, alternative='greater')

#set significance level
alpha = 0.05

#print results
print('Wilcoxon signed-rank test (one-sided):')
print('Statistic =', stat)
print('p-value =', p_value)
if p_val < alpha:
    print("Reject the null hypothesis. Classical art has higher preference ratings than modern art.")
else:
    print("Fail to reject the null hypothesis. Classical and modern art have similar preference ratings.")


#QUESTION #2

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.stats import ranksums
from scipy.stats import wilcoxon
import seaborn as sns
from scipy.stats import shapiro
import statsmodels.api as sm
from scipy.stats import levene

#import the dataset
art_data = pd.read_csv('theData.csv')
art_data.head()

#columns 36-70 represent modern art
modern_art = art_data.iloc[:, 35:70]
modern_art_list = modern_art.stack()
#columns 71-91 represent non-human art
non_human_art = art_data.iloc[:, 70:91]
non_human_art_list = non_human_art.stack()

plt.hist(modern_art_list, bins=15)
plt.show()

plt.hist(non_human_art_list, bins=15)
plt.show()

#comparative histogram
sns.histplot(data=modern_art_list, color='blue', alpha=0.5, label='Modern Pref. Ratings', bins=7, binrange=(0.5, 7.5))
sns.histplot(data=non_human_art_list, color='red', alpha=0.5, label='Non-Human Pref. Ratings', bins=7, binrange=(0.5, 7.5))

#labels and legend
plt.xlabel('Art Preference Ratings')
plt.ylabel('Frequency')
plt.xticks(np.arange(1, 8))
plt.legend(loc='upper left')

#plot
plt.show()

#test for non-normality - modern
stat, p = shapiro(modern_art_list)

#results
alpha = 0.05
if p > alpha:
    print('normal')
else:
    print('non-normal')

print("p-value: {:.4f}".format(p))

#test for non-normality - non-human
stat, p = shapiro(non_human_art_list)

#results
alpha = 0.05
if p > alpha:
    print('normal')
else:
    print('non-normal')

print("p-value: {:.10f}".format(p))

modern_column_means = modern_art.mean(axis=0)
modern_mean = modern_column_means.mean()
print(modern_mean)

modern_std = modern_art.stack().std()
print(modern_std)

non_human_column_means = non_human_art.mean(axis=0)
non_human_mean = non_human_column_means.mean()
print(non_human_mean)

non_human_std = non_human_art.stack().std()
print(non_human_std)

#wilcoxon rank sum test to adjust for non-normal data
t_stat, p_val = ranksums(modern_art_list, non_human_art_list)
alpha = 0.05

#print stats
print("wilcoxon statistic: ", t_stat)
print("p-value: ", p_val)

#conclusion
if p_val < alpha/2:
    print("Reject the null hypothesis. The median preference ratings for classical and modern art are significantly different.")
elif p_val > 1 - alpha/2:
    print("Reject the null hypothesis. The median preference ratings for classical and modern art are significantly different.")
else:
    print("Fail to reject the null hypothesis. The median preference ratings for classical and modern art are not significantly different.")

modern_median = np.median(modern_art_list)
non_human_median = np.median(non_human_art_list)

print("Median of modern_art_list:", modern_median)
print("Median of nonhuman_art_list:", non_human_median)


#QUESTION #3

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.stats import ranksums
from scipy.stats import wilcoxon
import seaborn as sns
from scipy.stats import shapiro
import statsmodels.api as sm
from scipy.stats import levene

#import the dataset
art_data = pd.read_csv('theData.csv')
art_data.head()

#first, create a dataset with just the preference ratings and the gender of the user
preference_ratings = art_data.iloc[:, :91]
user_gender = art_data[['gender']]
preference_and_gender = preference_ratings
preference_and_gender['gender'] = user_gender
preference_and_gender.head()

#then, it is necessary to remove all of the rows that don't include the gender of the user (row-wise removal of NaNs)
preference_and_gender = preference_and_gender.dropna()
preference_and_gender.head()
#find how many rows we had to remove due to missing gender
preference_and_gender.shape[0]

#now, we need to split the data into preference ratings of women and preference ratings of men
male_preferences = preference_and_gender[preference_and_gender['gender'] == 1]
female_preferences = preference_and_gender[preference_and_gender['gender'] == 2]

#then we have to drop the gender column so it doesn't interfere with the ratings
male_preferences = male_preferences.drop('gender', axis = 1)
female_preferences = female_preferences.drop('gender', axis = 1)

male_preferences.shape

female_preferences.shape

#turn the arrays into list for easier data manipulation
male_preferences_list = male_preferences.stack()
female_preferences_list = female_preferences.stack()

plt.hist(male_preferences_list, bins=15)
plt.show()

plt.hist(female_preferences_list, bins=15)
plt.show()

#test for normality of male preferences
stat, p = shapiro(male_preferences_list)

#results
alpha = 0.05
if p > alpha:
    print('normal')
else:
    print('non-normal')

print("p-value: {:.4f}".format(p))

#test for normality of female preferences
stat, p = shapiro(female_preferences_list)

#results
alpha = 0.05
if p > alpha:
    print('normal')
else:
    print('non-normal')

print("p-value: {:.4f}".format(p))

#comparative histogram
sns.histplot(data=male_preferences_list, color='blue', alpha=0.5, label='Male Pref. Ratings', bins=7, binrange=(0.5, 7.5))
sns.histplot(data=female_preferences_list, color='red', alpha=0.5, label='Female Pref. Ratings', bins=7, binrange=(0.5, 7.5))

#labels and legend
plt.xlabel('Art Preference Ratings')
plt.ylabel('Frequency (# of Ratings)')
plt.xticks(np.arange(1, 8))
plt.legend(loc='upper left')

#plot
plt.show()

#wilcoxon rank sum test
stat, p_value = ranksums(female_preferences_list, male_preferences_list, alternative='greater')

#significance level
alpha = 0.05

#results
print('Wilcoxon rank-sum test (one-sided):')
print('Statistic =', stat)
print('p-value =', p_value)
if p_value < alpha:
    print('Reject null hypothesis.')
else:
    print('Fail to reject null hypothesis.')


#QUESTION #4

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.stats import ranksums
from scipy.stats import wilcoxon
import seaborn as sns
from scipy.stats import shapiro
import statsmodels.api as sm
from scipy.stats import levene

#import the dataset
art_data = pd.read_csv('theData.csv')
art_data.head()

#first, create a dataset with just the preference ratings and the gender of the user
preference_ratings = art_data.iloc[:, :91]
user_art_education = art_data[['art_education']]
preference_and_arted = preference_ratings
preference_and_arted['art_education'] = user_art_education
preference_and_arted.head()

#then, it is necessary to remove all of the rows that don't include the art education of the user (row-wise removal of NaNs)
preference_and_arted = preference_and_arted.dropna()
preference_and_arted.head()
#find how many rows we had to remove due to missing gender
preference_and_arted.shape[0]

#now, we need to split the data into preference ratings of those with art education and those without
arted_preferences = preference_and_arted[preference_and_arted['art_education'].isin([1, 2, 3])]
no_arted_preferences = preference_and_arted[preference_and_arted['art_education'] == 0]

#then we have to drop the gender column so it doesn't interfere with the ratings
arted_preferences = arted_preferences.drop('art_education', axis = 1)
no_arted_preferences = no_arted_preferences.drop('art_education', axis = 1)

arted_preferences.head()

arted_preferences.shape

no_arted_preferences.shape

#turn the arrays into list for easier data manipulation
arted_preferences_list = arted_preferences.stack()
no_arted_preferences_list = no_arted_preferences.stack()
arted_preferences_list.shape

plt.hist(arted_preferences_list, bins=15)
plt.show()

plt.hist(no_arted_preferences_list, bins=15)
plt.show()

#comparative histogram
sns.histplot(data=arted_preferences_list, color='blue', alpha=0.5, label='Art Ed. Pref. Ratings', bins=7, binrange=(0.5, 7.5))
sns.histplot(data=no_arted_preferences_list, color='red', alpha=0.5, label='No Art Ed. Pref. Ratings', bins=7, binrange=(0.5, 7.5))

#labels and legend
plt.xlabel('Art Preference Ratings')
plt.ylabel('Frequency (# of Ratings)')
plt.xticks(np.arange(1, 8))
plt.legend(loc='upper left')

plt.ylim(0, max(max(np.histogram(arted_preferences_list, bins=7)[0]), max(np.histogram(no_arted_preferences_list, bins=7)[0])) + 10)


#plot
plt.show()

#wilcoxon rank sum test
statistic, p_value = ranksums(arted_preferences_list, no_arted_preferences_list, alternative='two-sided')

print("Wilcoxon rank-sum test statistic: ", statistic)
print("P-value: ", p_value)
if p_value < alpha:
    print("Reject the null hypothesis. The median preference ratings for classical and modern art are significantly different.")
else:
    print("Fail to reject the null hypothesis. The median preference ratings for classical and modern art are not significantly different.")


#QUESTION #5

import pandas as pd
import numpy as np
import math
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.stats import ranksums
from scipy.stats import wilcoxon
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from mord import LogisticAT

#import the dataset
art_data = pd.read_csv('theData.csv')
art_data.head()

#first, we create two datasets, one with the art preference ratings and one with the art energy ratings
preference_ratings = art_data.iloc[:, :91]
energy_ratings = art_data.iloc[:, 91:182]

#turn the energy ratings dataframe into one predictor variable
energy_ratings_np = np.concatenate(energy_ratings.values, axis=0)
energy_ratings_predictor = pd.Series(energy_ratings_np.ravel())

#turn the preference ratings dataframe into one outcome variable
preference_ratings_np = np.concatenate(preference_ratings.values, axis=0)
preference_ratings_outcome = pd.Series(preference_ratings_np.ravel())

energy_ratings_list = energy_ratings.stack()
preference_ratings_list = preference_ratings.stack()

plt.hist(energy_ratings_predictor, bins=15)
plt.show()

plt.hist(preference_ratings_outcome, bins=15)
plt.show()

#reshape
X.shape
r = X

#create a LinearRegression object
model = LinearRegression()

#fit the model on the data
model.fit(X, y)

#make predictions on the same data
y_pred = model.predict(X)

#calculate and print the RMSE and R-squared
rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)
print("Intercept:", model.intercept_)
print("Coefficient:", model.coef_[0])
print("RMSE: {:.2f}".format(rmse))
print("R-squared: {:.5f}".format(r2))

#cross validation using the k-fold method
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np

# define the number of splits (k)
k = 5

# define the number of samples in the dataset
n = len(X)

# initialize an array to store the evaluation scores
eval_scores = np.zeros(k)

# create a KFold object
kf = KFold(n_splits=k, shuffle=True, random_state=10700782)

# loop through each fold
for i, (train_index, test_index) in enumerate(kf.split(X)):
    
    # extract the training and test sets for this fold
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # fit a model on the training set
    model = sm.OLS(y_train, X_train).fit()
    
    # make predictions on the test set
    y_pred = model.predict(X_test)
    
    # calculate the root mean squared error and store it in the array
    eval_scores[i] = np.sqrt(mean_squared_error(y_test, y_pred))

# print the evaluation scores for each fold
for i, score in enumerate(eval_scores):
    print("RMSE for fold", i+1, ":", score)

# print the mean and standard deviation of the evaluation scores
print("Mean RMSE:", np.mean(eval_scores))
print("Standard deviation of RMSE:", np.std(eval_scores))

#plotting
n = len(r)

#set the size of the dots based on the number of data points
sizes = n * 0.5

#create a scatter plot with varying dot size
plt.scatter(r, y, color='blue', alpha=0.5, s=sizes)

#add the regression line
plt.plot(r, y_pred, color='red')

# set the x and y axis limits
plt.xlim([0, 7])
plt.ylim([0, 7])

#axis labels and a title
plt.xlabel("Preference Ratings")
plt.ylabel("Energy Ratings")
plt.title("Regression of Energy Ratings on Preference Ratings")
legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label='Alpha = 0.5', markerfacecolor='blue', markersize=10, alpha=0.5)]
plt.legend(handles=legend_elements)

#plot
plt.show()

#ordinal regression score
import pandas as pd
from mord import LogisticAT
from sklearn.preprocessing import StandardScaler
#applying feature scaling on the input data
scaler = StandardScaler()
X = scaler.fit_transform(X)
#creating the multinomial logistic regression model
regressor = LogisticAT(alpha=1.0, verbose=0)
regressor.fit(X, y)
#evaluating the score of the model
score = model.score(X, y)
print(score)


#QUESTION #6

#run regression in many different ways
#first, simply demogrpahic columns and energy ratings as predictors
#then, try standardizing
#then, try dimensionality reduction
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.stats import ranksums
from scipy.stats import wilcoxon
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
import math
from sklearn.ensemble import RandomForestRegressor

#import the dataset
art_data = pd.read_csv('theData.csv')
art_data.head()

#first, we create three datasets, one with the art preference ratings, one with the art energy ratings, and one with demographic information
preference_ratings = art_data.iloc[:, :91]
energy_ratings = art_data.iloc[:, 91:182]
demographic_data = art_data.iloc[:, 215:221]

#then, we need to combine these three, or just remove the columns that these three don't include, and remove the rows that have NaN values, as they are invalid
preference_energy_demographic = art_data.drop(art_data.columns[182:215], axis=1)
preference_energy_demographic.head()

#then, it is necessary to remove all of the rows that don't include all of the demographic information of the user
preference_energy_demographic1 = preference_energy_demographic.dropna()
preference_energy_demographic1.head()
#find how many rows we had to remove due to missing gender
preference_energy_demographic1.shape[0]
#21 rows dropped that do not contain this information

preference_energy_demographic1.head()

#now i need to take this dataframe and make the preference ratings one column, the energy ratings one column, and all of the demographic information seperate columns

#turn the preference ratings dataframe into one outcome variable by reducing each of the 91 rows to means
preference_medians = preference_energy_demographic1.iloc[:, :91].median(axis=1)
energy_medians = preference_energy_demographic1.iloc[:, 91:182].median(axis=1)

#make the six demographic columns their own predictor variables for consistency
age_predictor = preference_energy_demographic1['age']
gender_predictor = preference_energy_demographic1['gender']
political_orientation_predictor = preference_energy_demographic1['political_orientation']
art_education_predictor = preference_energy_demographic1['art_education']
general_sophistication_predictor = preference_energy_demographic1['general_sophistication']
artist_predictor = preference_energy_demographic1['artist']

#create a new DataFrame with the predictor and outcome variables
#predictors being energy ratings, and the six demographic information columns, and the output being the preference ratings
ratings_data = pd.DataFrame({'energy_predictor': energy_medians,
                             'age_predictor': age_predictor,
                             'gender_predictor': gender_predictor,
                             'poli_ori_predictor': political_orientation_predictor,
                             'art_edu_predictor': art_education_predictor,
                             'gen_soph_predictor': general_sophistication_predictor,
                             'artist_predictor': artist_predictor,
                             'preference_outcome': preference_medians})

ratings_data.head()
ratings_data.shape

ratings_data.head(300)

#create X and y variables
X = ratings_data.iloc[:, :7]  # predictor variables (first 7 columns)
y = ratings_data['preference_outcome']  # outcome variable (last column)

#add a constant term to the predictor variables
# create a LinearRegression object
model = LinearRegression()

# fit the model on the data
model.fit(X, y)

# make predictions on the same data
y_pred = model.predict(X)

# calculate and print the RMSE and R-squared
rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)
print("Intercept:", model.intercept_)
print("Coefficient:", model.coef_)
print("RMSE: {:.2f}".format(rmse))
print("R-squared: {:.5f}".format(r2))

from sklearn.metrics import mean_squared_error, r2_score

# calculate RMSE
rmse = np.sqrt(mean_squared_error(y, y_pred))
print('RMSE:', rmse)

# calculate R-squared
r2 = r2_score(y, y_pred)
print('R-squared:', r2)

#residual plot for multiple linear regression
y_pred = model.predict(X)
residuals = y - y_pred

# Plot the residuals against the predicted values
plt.scatter(y_pred, residuals)
plt.axhline(y=0, color='r', linestyle='-')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.title('Residual plot for multiple linear regression')
plt.show()

#lasso regression with alpha of 0.01
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
#split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10700782)

#standardize the predictor variables
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#perform Lasso regression
lasso = Lasso(alpha=0.01)
lasso.fit(X_train_scaled, y_train)

#make predictions on the test set
y_pred = lasso.predict(X_test_scaled)

#evaluate the model performance
r_squared = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

#print the results
print("RMSE:", rmse)

#plot with alpha of 0.01
coefficients = lasso.coef_

#plot the coefficients
plt.figure(figsize=(10, 6))
plt.bar(range(len(coefficients)), coefficients)
plt.xticks(range(len(coefficients)), X.columns, rotation=90)
plt.xlabel('Predictor Variables')
plt.ylabel('Coefficient')
plt.title('LASSO Regression Coefficients')
plt.show()

#lasso regression with alpha of 0.1
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
#split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10700782)

#standardize the predictor variables
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#perform Lasso regression
lasso = Lasso(alpha=0.1)
lasso.fit(X_train_scaled, y_train)

#make predictions on the test set
y_pred = lasso.predict(X_test_scaled)

#evaluate the model performance
r_squared = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

#print the results
print("RMSE:", rmse)

#plot with alpha of 0.1
coefficients = lasso.coef_

#plot the coefficients
plt.figure(figsize=(10, 6))
plt.bar(range(len(coefficients)), coefficients)
plt.xticks(range(len(coefficients)), X.columns, rotation=90)
plt.xlabel('Predictor Variables')
plt.ylabel('Coefficient')
plt.title('LASSO Regression Coefficients')
plt.show()


#QUESTION 7

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.stats import ranksums
from scipy.stats import wilcoxon
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

#import the dataset
art_data = pd.read_csv('theData.csv')
art_data.head()

#get the means of each row
pd.set_option('display.max_rows', None)  # Set option to display all rows
pd.set_option('display.max_columns', None)  # Set option 
preference_means = art_data.iloc[:, :91].mean()
energy_means = art_data.iloc[:, 91:182].mean()
print(preference_means)

print(energy_means)

preference_means_list = preference_means.tolist()
energy_means_list = energy_means.tolist()

#new data matrix of preference and energy means
ratings_data = pd.DataFrame({'preference_means': preference_means_list,
                             'energy_means': energy_means_list})
print(ratings_data)

#silhouette method
#define a range of k values to try
k_values = range(2, 30)

#initialize an empty list to store the silhouette scores for each k
silhouette_scores = []

#loop over each k value and compute the corresponding silhouette score
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=10700782)
    kmeans.fit(ratings_data)
    labels = kmeans.labels_
    score = silhouette_score(ratings_data, labels)
    silhouette_scores.append(score)

#find the index of the highest silhouette score
best_index = silhouette_scores.index(max(silhouette_scores))

#get the best k value
best_k = k_values[best_index]

#print the best k value and corresponding silhouette score
print(f"Best k value: {best_k}")
print(f"Silhouette score: {silhouette_scores[best_index]}")

#now run k-means with decided on k
#define the number of clusters
k = 3

#initialize k-means model
kmeans = KMeans(n_clusters=k, random_state=10700782)

#fit the model to your data
kmeans.fit(ratings_data)

#final cluster assignments for each data point
labels = kmeans.labels_

#final cluster centroids
centroids = kmeans.cluster_centers_

#now plotting the results
#convert ratings_data to a numpy array
ratings_array = ratings_data.values

# Define the colors for each cluster
colors = ['red', 'blue', 'purple']

#clusters
for i in range(k):
    #data points assigned to this cluster
    cluster_data = ratings_array[labels == i]
    #data points with the corresponding color
    plt.scatter(cluster_data[:, 0], cluster_data[:, 1], color=colors[i])

#final cluster centroids
plt.scatter(centroids[:, 0], centroids[:, 1], color='black', marker='x', s=100)

#plot - title and axis labels
plt.title('K-Means Clustering for 91 Art Pieces')
plt.xlabel('Preference Means')
plt.ylabel('Energy Means')

#show lot
plt.show()


#QUESTION #8

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.stats import ranksums
from scipy.stats import wilcoxon
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, roc_curve, r2_score
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import requests
from PIL import Image
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LogisticRegression

np.random.seed(10700782)

#import the dataset
data = pd.read_csv('theData.csv')
data.head()

cols_to_keep = list(range(92)) + list(range(206, 216))
data = data.iloc[:, cols_to_keep]

#make the 91 columns the mean art preferences, in one variable
self_image_data = art_data.iloc[:, 205:215]
self_image_data.head()

#checking for accuracy
self_image_data.shape

#now we need to drop rows with NaN values, as per usual
self_image_data = self_image_data.dropna()
self_image_data.head()

#checking for accuracy
self_image_data.shape

#image plot
aspect_ratio = self_image_data.shape[1] / self_image_data.shape[0]

#set the extent of the image to make the columns wider
extent = [0, self_image_data.shape[1], 0, self_image_data.shape[0]]

#create the plot
plt.imshow(self_image_data, extent=extent, aspect=aspect_ratio)
plt.colorbar()
plt.title('User Ratings for 10 Self-Image Statements')
plt.xlabel('Statement')
plt.ylabel('User')

plt.show()

#correlation matrix
corrMatrix = np.corrcoef(self_image_data, rowvar=False)
plt.imshow(corrMatrix)
plt.colorbar()
plt.title('Correlation Matrix of Self-Image Statements')
plt.xlabel('Statement')
plt.ylabel('Statement')

#z score the data
zscoredData = stats.zscore(self_image_data)

#initialize PCA object and fit our data
pca = PCA().fit(zscoredData)

#single vector of eigenvalues in descneding order of magnitude
eigVals = pca.explained_variance_
print(eigVals)

#loadings [eigenvectors] = weights per factor in terms of the original data.
loadings = pca.components_
print(loadings)

#rotate data - we had 300 users and 11 columns (variables) - 300 columns and 11 row
rotatedData = pca.fit_transform(zscoredData)
print(rotatedData)

#eigvals in terms of var explained
varExplained = eigVals/sum(eigVals)*100

for ii in range(len(varExplained)):
    print(varExplained[ii].round(3))

#scree plot
numQuestions = 10
x = np.linspace(1, numQuestions, numQuestions)
plt.bar(x, eigVals, color="grey")
plt.plot([0.5, numQuestions], [1,1], color='orange')
plt.xlabel('Principle Component')
plt.ylabel('Eigenvalue')
plt.title('Scree Plot')
plt.show()

#reach into loadings matrix and look at each question per PC
whichPrincipleComponent = 1
plt.bar(x, loadings[whichPrincipleComponent,:]*-1)
plt.xlabel('Question')
plt.ylabel('Loading')
plt.show()

#displaying data of first PCA
plt.plot(rotatedData[:,1]*-1, 'o', markersize=5)
plt.show()

#displayu first principle component
pc1 = rotatedData[:, 1]*-1
print(pc1)
pc_predictor = pc1

#now i need to get the art preference ratings in a seperate variable as the outcome, in order to run regression of the principle component on the art preferences
art_preferences = art_data.iloc[:, :91]
art_preferences1 = art_data.iloc[:, 205:215]
result = pd.concat([art_preferences, art_preferences1], axis=1, join='inner')
result.head()

result.shape

new_result = result.dropna()

art_preferences_outcome = new_result.iloc[:, :91]
art_preferences_outcome.head()

art_preferences_outcome.shape

art_preference_medians = np.median(art_preferences_outcome, axis=1)
art_preference_medians

pc_predictor.shape

#preparing for linear regression
X = np.array(pc_predictor).reshape(-1, 1)
y = np.array(art_preference_medians).reshape(-1, 1)

# create a LinearRegression object
model = LinearRegression()

# fit the model on the data
model.fit(X, y)

# make predictions on the same data
y_pred = model.predict(X)

# calculate and print the RMSE and R-squared
rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)
print("Intercept:", model.intercept_)
print("Coefficient:", model.coef_[0])
print("RMSE: {:.2f}".format(rmse))
print("R-squared: {:.2f}".format(r2))

#linear regression scatter plot
plt.scatter(X, y, color='blue', alpha=0.5)

#plot the regression line
plt.plot(X, y_pred, color='red')

#add axis labels and a title
plt.xlabel('PC1')
plt.ylabel('Art Preference Medians')
plt.title('Linear Regression Model')

#add a text box with the RMSE and R-squared values
textstr = 'RMSE = {:.2f}\nR-squared = {:.2f}'.format(rmse, r2)
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
        verticalalignment='top', bbox=props)

#show the plot
plt.show()


#QUESTION #9

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.stats import ranksums
from scipy.stats import wilcoxon
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import requests
from PIL import Image
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

#import the dataset
art_data1 = pd.read_csv('theData.csv')
art_data1.head()

#now we need to drop rows with NaN values, as per usual
art_data = art_data1.dropna()
art_data.head()

dark_data = art_data.iloc[:, 182:194]
dark_data.head()

dark_data.shape

#imagw plot
aspect_ratio = dark_data.shape[1] / dark_data.shape[0]

# Set the extent of the image to make the columns wider
extent = [0, dark_data.shape[1], 0, dark_data.shape[0]]

# Create the plot
plt.imshow(dark_data, extent=extent, aspect=aspect_ratio)
plt.colorbar()
plt.title('User Ratings for 12 Dark Personality Statements')
plt.xlabel('Statement')
plt.ylabel('User')

plt.show()

#correlation matrix
corrMatrix = np.corrcoef(dark_data, rowvar=False)
plt.imshow(corrMatrix)
plt.title('Correlation Matrix of Dark Personality Statements')
plt.xlabel('Statement')
plt.ylabel('Statement')
plt.colorbar()

#z scored data
zscoredData = stats.zscore(dark_data)

#initialize PCA object and fit our data
pca = PCA().fit(zscoredData)

#single vector of eigenvalues in descneding order of magnitude
eigVals = pca.explained_variance_
print(eigVals)

#loadings [eigenvectors] = weights per factor in terms of the original data.
loadings = pca.components_
print(loadings)

#rotate data - we had 300 users and 12 columns (variables) - 300 columns and 12 rows
rotatedData = pca.fit_transform(zscoredData)
print(rotatedData)

#eigvals in terms of var explained
varExplained = eigVals/sum(eigVals)*100

for ii in range(len(varExplained)):
    print(varExplained[ii].round(3))

#scree plot
numQuestions = 12
x = np.linspace(1, numQuestions, numQuestions)
plt.bar(x, eigVals, color="grey")
plt.plot([0.5, numQuestions], [1,1], color='orange')
plt.xlabel('Principle Component')
plt.ylabel('Eigenvalue')
plt.title('Scree Plot')
plt.show()

#reach into loadings matrix and look at each component
whichPrincipleComponent = 1
plt.bar(x, loadings[whichPrincipleComponent,:]*-1)
plt.xlabel('Statement')
plt.ylabel('Loading')
plt.title('PC1')
plt.show()

#now for component two
whichPrincipleComponent = 2
plt.bar(x, loadings[whichPrincipleComponent,:]*-1)
plt.xlabel('Statement')
plt.title('PC2')
plt.ylabel('Loading')
plt.show()

#and now component three
whichPrincipleComponent = 3
plt.bar(x, loadings[whichPrincipleComponent,:]*-1)
plt.xlabel('Statement')
plt.ylabel('Loading')
plt.title('PC3')
plt.show()

art_preferences = art_data.iloc[:, :91]
art_preferences.head()
art_preference_outcome = art_preferences.median(axis=1)
art_preference_outcome.head()
y = art_preference_outcome

#examining pc1
pc1 = rotatedData[:, 1]*-1
print(pc1)
pc_predictor = pc1

#now pc2
pc2 = rotatedData[:, 2]*-1
print(pc1)
pc_predictor = pc2

#now pc3
pc3 = rotatedData[:, 3]*-1
print(pc3)
pc_predictor = pc3

X = np.concatenate((pc1, pc2, pc3)).reshape(-1, 3)
print(X.shape)

#linear regression
model = LinearRegression()

# fit the model on the data
model.fit(X, y)

# make predictions on the same data
y_pred = model.predict(X)

# calculate and print the RMSE and R-squared
rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)
print("Intercept:", model.intercept_)
print("Coefficient:", model.coef_)
print("RMSE: {:.2f}".format(rmse))
print("R-squared: {:.2f}".format(r2))


#QUESTION #10

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.stats import ranksums
from scipy.stats import wilcoxon
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import requests
from PIL import Image
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

#import the dataset
art_data1 = pd.read_csv('theData.csv')
art_data1.head()

#row-wise dropping of NaNs
art_data = art_data1.dropna()

#making the political orientation column binary - making leftist 0, and anything else 1
art_data.loc[art_data['political_orientation'] <= 2, 'political_orientation'] = 0
art_data.loc[art_data['political_orientation'] > 2, 'political_orientation'] = 1
art_data.head()

#creating variables for regression
X = pd.concat([art_data.iloc[:, :217], art_data.iloc[:, 218:]], axis=1)
y = art_data.iloc[:, 217]

X.head()

#now to perform logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

#split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10700782)

# Create and fit the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

#predictions on the test set
y_pred = model.predict(X_test)

#accuracy and AUC score
accuracy = accuracy_score(y_test, y_pred)
auc_score = roc_auc_score(y_test, y_pred)

#plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

#print accuracy and AUC score
print("Accuracy:", accuracy)
print("AUC score:", auc_score)

