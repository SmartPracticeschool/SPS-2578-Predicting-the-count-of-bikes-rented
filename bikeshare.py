# Count the rented bikes

# Importing necessary libraries
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sklearn.metrics as met
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt

# Function to convert 2d list to 1d list
def twoD_List_to_oneD_List(s):
    # initialize an empty 1d list
    lis1 = [ ]
    for i in range(len(s)):
        val = s[i][0]
        lis1.append(val)

    # return string   
    return lis1

# Z-Score method
def Z_Score(x):
    # Z-Score Outlier detection and removal method
    # Initialising & defining constants
    THRESHOLD = 3  # Setting the threshold for value so obtained

    finalData = {}

    print(x.info(verbose=True))
    # print(x.values)
    # print(x.shape)

    for _i in range(len(x.columns.values)):

        # getting the first column of independent variable
        column_name = x.columns[_i]

        # converting nd.array into 2D list
        arr = x[[column_name]].values
        list1 = arr.tolist()

        # return from function twoD_List_to_oneD_List()
        data = twoD_List_to_oneD_List(list1)
        # print(data)

        # Obtaining mean & standard deviation
        mean = np.mean(data) 
        std = np.std(data) 

        outlier = []        # empty list for outlier
        outlier_check = []  # emptty list for rechecking of outliers

        # Detecting the outliers
        for i in data: 
            z = (i-mean)/std 
            if z > THRESHOLD: 
                outlier.append(i)

        # if outlier == []:
        #     print("No outlier")
        # else:
        #     print('outlier in dataset is', outlier) 

        # Eliminating the outliers
        for i in outlier:
            for n,i_ in enumerate(data):
                if i == i_:
                    data[n] = round(mean)
        
        # Peforming a recheck on the outliers
        for i in data: 
            z = (i-mean)/std 
            if z > THRESHOLD: 
                outlier_check.append(i)

        # if outlier != []:
        #     print('outlier in dataset after eliminating: ', outlier_check)

        finalData[x.columns[_i]] = data

    df = pd.DataFrame(data=finalData)
    # print(df.info(verbose=True))
    return df.values

# Importing the dataset bikeshare.csv
dataset = pd.read_csv("bikeshare.csv")
# print(dataset.info(verbose=True))   # Seeing the information of dataset so loaded

# Check if any value is null in dataset
print(dataset.isnull().any())

# Obtaining independent and dependent variable
x = dataset.iloc[:,2:16]
y = dataset.iloc[:,16].values
# print(x)

# Performing Exploratory Data Analaysis
abcissa = [i for i in range(1,len(y)+1)]
fig=plt.figure()
ax=fig.add_axes([0,0,1,1])
ax.bar(abcissa, y, color='r')
ax.set_xlabel('Values')
ax.set_ylabel('Count of bikes rented')
ax.set_title('Bar Plot')
plt.show()

# Performing Z-Score to get rid of outliers
X = Z_Score(x)

# Splitting the model into training and testing 
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.4,random_state=0)

# Feature Scaling 
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Applying Multi-Linear Regression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

# Predicting the Test Results
Y_pred__ = regressor.predict(X_test)
Y_pred_ = abs(Y_pred__)
Y_pred = [ int(i) for i in Y_pred_]
print(Y_pred)
