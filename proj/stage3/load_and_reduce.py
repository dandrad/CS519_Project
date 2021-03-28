import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import re

def timestr_to_days(s):
    s = str(s)
    match = re.search('([\w.-]+) ([\w.-]+)', s)
    if match:
        x = str(match.group(1))
        y = str(match.group(2))
        multiplier = 1
        if "year" in y:
            multiplier = 365
        elif "month" in y:
            multiplier = 365 / 12
        elif "week" in y:
            multiplier = 7
        elif "day" in y:
            multiplier = 1
        return int(x) * multiplier


def load_and_reduce(intake_path, outcome_path, coding = "onehot", scale = True, DimRed = None):

    intake = pd.read_csv(intake_path,header=0)
    outcome = pd.read_csv(outcome_path,header=0)

    #drop columns we don't want
    intake.drop(['Name', 'MonthYear', 'Found Location'], axis=1, inplace = True)

    #drop animals we dont want
    animals = ['Other', 'Bird', 'Livestock']
    for i in animals:
        intake.drop(intake[intake['Animal Type']==i].index, inplace = True)

    intake['DateTime_in']=pd.to_datetime(intake['DateTime_in'])

    #drop outcome columns that we dont want or are repeat
    outcome = outcome.drop(['Name', 'MonthYear', 'Animal Type', 'Breed', 'Color', 'Outcome Subtype'], axis=1)
    outcome['DateTime_out']=pd.to_datetime(outcome['DateTime_out'])


    #sort datasets and drop duplicates
    intake = intake.sort_values(by=['DateTime_in'])
    intake = intake.drop_duplicates(subset=['Animal ID'], keep = 'last')

    outcome = outcome.sort_values(by=['DateTime_out'])
    outcome = outcome.drop_duplicates(subset=['Animal ID'], keep = 'last')

    # Merge the data sets on the animal ID and base the merge on the Outcome Data Set
    # So we don't have animals that are still in the shelter
    data = pd.merge(intake, outcome, on = 'Animal ID', how = 'right')

    data['Time in Shelter'] = data['DateTime_out']-data['DateTime_in']

    # Lets drop 'Age upon Intake' and 'Date of Birth' since these are represented in 'Age upon Outcome' and 'Time in Shelter'
    data = data.drop(['Age upon Intake', 'Date of Birth'], axis=1)

    #convert time in shelter to days
    data["time_in_shelter_days"] = data["Time in Shelter"]  / np.timedelta64(1,'D')

    #drop rows with NA
    data = data.dropna()

    #drop rows with negative time in shelter (due to bad data)
    negative_times = data[ data["time_in_shelter_days"] < 0].index
    data.drop(negative_times, inplace = True)
    data = data.drop(['DateTime_out', 'DateTime_in', 'Animal ID'], axis=1)


    # Create new column that has the animal age in days and drop 'Age Upon Outcome'
    data['Age (Days)'] = [timestr_to_days(datum) for datum in data['Age upon Outcome']]
    data = data.drop(['Age upon Outcome',"Time in Shelter"], axis=1)

    X = data.drop(['time_in_shelter_days'], axis = 1)
    y = data['time_in_shelter_days']


    #do onehot encoding for reasonable columns, leave out breed and color because
    #they have too many unique values
    if(coding == "onehot"):

        one_hot_cols = ["Intake Type","Intake Condition", "Animal Type","Outcome Type","Sex upon Outcome", "Sex upon Intake"]
        drop_cols = ["Breed","Color"]

        for col in one_hot_cols:

            print(col)
            print(len(X[col].unique()))
            print()

            dum_df = pd.get_dummies(X[col], columns=[col], prefix=col )

            X = X.join(dum_df)

            X.drop(col,axis = 1, inplace = True)

        X.drop(drop_cols, axis = 1, inplace = True)

    elif(coding == "label"):

        le = preprocessing.LabelEncoder()

        cols = ['Intake Type','Intake Condition', 'Animal Type', 'Sex upon Intake','Breed','Color','Outcome Type','Sex upon Outcome']
        for i in cols:
            X[i]=le.fit_transform(X[i])

    if(DimRed == "PCA"):

        pca = PCA(n_components = 5)

        pca.fit(X)

        X = pca.transform(X)

    if(scale == True and DimRed == "None"):

        scaled_features = X.copy()

        col_names = ['Age (Days)']
        features = scaled_features[col_names]
        scaler = StandardScaler().fit(features.values)
        features = scaler.transform(features.values)

        scaled_features[col_names] = features

        X = scaled_features


    return X, y
