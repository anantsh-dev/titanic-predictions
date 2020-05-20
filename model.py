import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV

#print basic stats for df
def df_stats(df, name="Train", label="Survived"):
    print("Stats for", name, "Data")
    print("Shape :", df.shape)
    print("Duplications", df.duplicated().sum())
    print("\nNull Values")
    print(df.isnull().sum())
    print("\nDescription")
    print(df.describe())
    print("\nSample Values")
    print(df.head())
    if name.lower()=="train":
        print("Target Distribution")
        print(df[label].value_counts())


#print correlation of survival wrt to input labels
def correlation_wrt_labels(df,labels=[]):
    for label in labels:
        print("The percentage of survived with respect to", label)
        print(100 * df.groupby(label).Survived.mean())


#fill missing values in df
def fill_missing(df):
    #one missing data in test Fare
    df["Fare"] = df["Fare"].fillna(-1)
    #only 2 vals missing so filled with mode
    df["Embarked"]=df["Embarked"].fillna(df["Embarked"].mode()[0])
    #create seperate class for missing Ages
    df["Age"] = df["Age"].fillna(-1)
    return df


#update names to grab prefix from names then one hot
def update_names(df):
    df["Name"] = df["Name"].apply(lambda x : x.split(", ")[1].split(".")[0])
    df["Name"] = df["Name"].replace(["Ms", "Mlle"], "Miss")
    df["Name"] = df["Name"].replace(["Sir"], "Mr")
    df["Name"] = df["Name"].replace(["Mme"], "Mrs")
    df["Name"] = df["Name"].apply(lambda x: "Other" if x not in {"Mr", "Miss", "Master", "Mrs"} else x)
    return df


#age to categories
def age_to_cat(df):
    cut_values=[-2, 0, 3, 12, 19, 35, 60, 80]
    label_names=["Missing", "Infants", "Children", "Teenagers", "Millennials", "GenX", "Senior"]
    df["Age"]=pd.cut(df["Age"], bins=cut_values, labels=label_names)
    return df


#join siblings and parent to form family
def sp_to_fam(df):
    df["Family"] = df.pop("SibSp") + df.pop("Parch")
    cut_values=[-1, 0, 3, 7, 15]
    label_names=["Alone", "Nuclear", "Joint", "Extended"]
    df["Family"]=pd.cut(df["Family"], bins=cut_values, labels=label_names)
    return df


def get_fare_bins(fares): return np.quantile(fares,[0,0.25,0.5,0.75,1])


def fare_to_cat(df, cut_values):
    label_names=["Staff", "Low", "Medium", "High"]
    df["Fare"]=pd.cut(df["Fare"], bins=cut_values, labels=label_names,include_lowest=True)
    return df


def remove_cols(df, cols=[]): return df.drop(cols, axis=1)


def init_xgb():
    xgb = XGBClassifier(objective='binary:logistic')
    params = {"n_estimators" : [200, 500, 1000],
                "learning_rate" : [0.01, 0.02, 0.05, 0.10, 0.20, 0.30] ,
                "max_depth" : [ 3, 4, 5, 6, 8, 9],
                "min_child_weight" : [ 1, 3, 5, 7 ,10],
                'subsample' : [0.6, 0.8, 1.0],
                "gamma" : [0.0, 0.5, 1, 1.5, 2, 5],
                "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7, 1.0 ] }
    return xgb, params


def get_predictions(X_train,y , X_test):
    model, params = init_xgb()
    skf = StratifiedKFold(n_splits=5, shuffle = True, random_state = 1001)
    random_search = RandomizedSearchCV(model, param_distributions=params, n_iter=50, 
        scoring='roc_auc', n_jobs=4, cv=skf.split(X_train,y), verbose=3, random_state=1001)
    random_search.fit(X_train,y)

    return random_search.predict(X_test)


def write_submission(ids, preds): 
    pd.DataFrame({'PassengerId': ids, 'Survived': preds}).to_csv('data/submission.csv', index=False)


#Col Names --> PassengerId Survived Pclass Name Sex Age SibSp Parch Ticket Fare Cabin Embarked
if __name__ == "__main__":
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    df_stats(train, name="Train")
    df_stats(test, name="Test")
    correlation_wrt_labels(train, labels=["Sex", "SibSp", "Parch", "Pclass"])
    train = fill_missing(train)
    test = fill_missing(test)
    train = update_names(train)
    test = update_names(test)
    train = age_to_cat(train)
    test = age_to_cat(test)
    correlation_wrt_labels(train, labels=["Age"])
    train = sp_to_fam(train)
    test  = sp_to_fam(test)
    fare_bin_values = get_fare_bins(train["Fare"])
    train = fare_to_cat(train, fare_bin_values)
    test = fare_to_cat(test, fare_bin_values)
    train = remove_cols(train, ["PassengerId","Ticket", "Cabin"])
    test = remove_cols(test, ["Ticket", "Cabin"])
    y = train.pop("Survived")
    features = ["Pclass", "Name", "Sex", "Age", "Family", "Fare", "Embarked"]
    X_train = pd.get_dummies(train,columns=features)
    test_ids = test.pop("PassengerId")
    X_test = pd.get_dummies(test,columns=features)
    y_preds = get_predictions(X_train,y, X_test)
    write_submission(y_preds)
