import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import re

#converts a time string to days
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


def load_and_reduce(intake_path, outcome_path, coding = "onehot", scale = True, DimRed = None, n_components = 1):
    
    '''
    Loads and processes raw data

            Parameters:
                    intake_path (str): Path to Animal Shelter Intake data
                    outcome_path (str): Path to Animal Shelter Outcome data
                    coding (str): Type of encoding used, either onehot, label, or None
                    scale (True/False): Boolean flag to control if scaling of Age occurs 
                    DimRed (str): Type of Dimension Reduction used, either PCA or None 
                    n_componets (int): If Dimension Reduction is used, how many components to keep

            Returns:
                    X (pd dataframe): Processed feature data
                    y (pd dataframe): Target data
    '''

    #load data
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
    #data = data.drop(['DateTime_out', 'DateTime_in'], axis=1)

    # Create new column that has the animal age in days and drop 'Age Upon Outcome'
    data['Age (Days)'] = [timestr_to_days(datum) for datum in data['Age upon Outcome']]
    data = data.drop(['Age upon Outcome',"Time in Shelter"], axis=1)
    
    
    # Feature Engineering
    print("Begin Feature Engineering\n")
    
    
    # Get Cardinality of Breed and Color, print
    start_breed_card = len(data["Breed"].unique())
    start_color_card = len(data["Color"].unique())
    print("Starting Cardinality of Breed",start_breed_card)
    print("Starting Cardinality of Color",start_color_card)
    
    # Create new columns that denote animals with Mixed breed or color
    data['Mix_Breed'] = [1 if (("Mix" in x) or ("/" in x)) else 0 for x in data['Breed']]
    data['Mix_Color'] = [1 if ("/" in x) else 0 for x in data['Color']]
    
    # Reduce the Breed column, assume first breed is the dominate breed
    data['Breed'] = [x.split('/')[0] for x in data['Breed']]
    data["Breed"] = [x.split(' Mix')[0] for x in data['Breed']]
   
    # Reduce the Color column, assume first color is the dominate color
    data['Color'] = [x.split('/')[0] for x in data['Color']]
        
    # Get new Cardinality of Breed and Color, print
    redmix_breed_card = len(data["Breed"].unique())  
    redmix_color_card = len(data["Color"].unique())
    print("\nCardinality of Breed After Removing Mix and /",redmix_breed_card)
    print("Cardinality of Color After Removing /",redmix_color_card)    
        
   
    # Introduce Sex_Changed column, 1 if Sex is different between income/outcome, 0 otherwise      
    data['Sex_Changed'] = data['Sex upon Intake'] != data['Sex upon Outcome']       
    data['Sex_Changed'] = np.where(data["Sex_Changed"] == True, 1, 0)
            
      
    # Boil down rare breeds < 20 instances, to common "Ultra Rare" breed, 20-50 instances to "Rare" bree
    breeds = data["Breed"] 
    counts = breeds.value_counts()
    ultra_rares = counts[counts < 20]
    rares = counts[counts <= 50]

    
    rares = rares[~rares.index.isin(ultra_rares.index)]
    
    ultra_rare_breeds = np.asarray(ultra_rares.index)
    rare_breeds = np.asarray(rares.index)
    
    data['Breed'] = ["Rare" if (x in rare_breeds) else x for x in data['Breed']]
    data['Breed'] = ["Ultra Rare" if (x in ultra_rare_breeds) else x for x in data['Breed']]
  
    # Boil down rare colors < 100 instances, to "Rare" color
    colors = data["Color"] 
    counts_color = colors.value_counts()
    rares_color = counts_color[counts_color <= 100]
    
    rare_colors = np.asarray(rares_color.index)
    
    data['Color'] = ["Rare" if (x in rare_colors) else x for x in data['Color']]
    
    
    redrare_breed_card = len(data["Breed"].unique())  
    redrare_color_card = len(data["Color"].unique())
    print("\nCardinality of Breed After Boiling Down Rare Breeds",redrare_breed_card)
    print("Cardinality of Color After Boiling Down Rare Colors",redrare_color_card)  
    
   
    X = data.drop(['time_in_shelter_days'], axis = 1)
    y = data['time_in_shelter_days']

    print("\nTotal Dimensions of X before Encoding", X.shape)
    

    #do onehot encoding 
    if(coding == "onehot"):
        #columns you want to encode
        one_hot_cols = ["Intake Type","Intake Condition", "Animal Type","Outcome Type","Sex upon Outcome", "Sex upon Intake", "Breed","Color"]
        
        #columns you want to drop (if any)
        drop_cols = []

        print("\nEncoding Columns", one_hot_cols)
        print("Droping Columns", drop_cols)
        print()      
        for col in one_hot_cols:

            
            print(col,"Cardinality - ", len(X[col].unique()))
            print()

            dum_df = pd.get_dummies(X[col], columns=[col], prefix=col )

            X = X.join(dum_df)

            X.drop(col,axis = 1, inplace = True)

        X.drop(drop_cols, axis = 1, inplace = True)

    #do label encoding
    elif(coding == "label"):

        le = preprocessing.LabelEncoder()

        cols = ['Intake Type','Intake Condition', 'Animal Type', 'Sex upon Intake','Breed','Color','Outcome Type','Sex upon Outcome']
        for i in cols:
            X[i]=le.fit_transform(X[i])

    #do PCA
    if(DimRed == "PCA"):
        
        print("Performing PCA")
     
        pca = PCA(n_components = n_components)
        pca.fit(X)
        X = pca.transform(X)
        
        print("Keeping n_components =",n_components)
        print("PCA Explained Variance Ratio - ", pca.explained_variance_ratio_)
            
    #scale, if desired, and didn't already reduce dimensions
    if(scale == True and DimRed == "None"):

        scaled_features = X.copy()

        col_names = ['Age (Days)']
        print("Scaling", col_names)
        
        features = scaled_features[col_names]
        scaler = StandardScaler().fit(features.values)
        features = scaler.transform(features.values)

        scaled_features[col_names] = features

        X = scaled_features

    print("Final Dimensions of X", X.shape)
        
    return X, y
