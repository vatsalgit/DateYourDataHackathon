# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn import cross_validation
from sklearn.feature_selection import RFE
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.ensemble import BaggingClassifier

#Reading the raw data files 
train_raw = pd.read_csv("trainfiles/traincsv/train.csv")
test_raw = pd.read_csv('test-date-your-data/test.csv')
y_train = train_raw.Is_Shortlisted
merged = pd.concat([train_raw,test_raw])
internship = pd.read_csv("trainfiles/Internship/Internship.csv")
student = pd.read_csv("trainfiles/Student/student.csv")
#Removing duplicates from the student file 
student = student.drop_duplicates("Student_ID")
#Merging all data to evaluate features and train the data 
merged = pd.merge(merged,student, on='Student_ID',how='left')
merged = pd.merge(merged,internship,on='Internship_ID',how='left')

merged = merged.drop('Is_Shortlisted',axis=1)


len(merged.columns)
merged.Percentage_PG = merged.Performance_PG/merged.PG_scale

merged.Percentage_UG = merged.Performance_UG/merged.UG_Scale

merged['Percentage_PG']=merged.Percentage_PG
merged['Percentage_UG']=merged.Percentage_UG

merged = merged.drop(merged.columns[[14,15,16,17]], axis=1) 

            
#Converting categorical variables to numeric attributes 
merged = merged.fillna('NA')         
le = preprocessing.LabelEncoder()
merged['Expected_Stipend'] = le.fit_transform(merged.Expected_Stipend)
merged['Preferred_location'] = le.fit_transform(merged.Preferred_location)
merged['Institute_Category'] = le.fit_transform(merged.Institute_Category)
merged['Institute_location'] = le.fit_transform(merged.Institute_location)
merged['hometown'] = le.fit_transform(merged.hometown)
merged['Degree'] = le.fit_transform(merged.Degree)
merged['Current_year'] = le.fit_transform(merged.Current_year)
merged['Experience_Type'] = le.fit_transform(merged.Experience_Type)
merged['Location'] = le.fit_transform(merged.Location)
merged['Internship_Type'] = le.fit_transform(merged.Internship_Type)
merged['Internship_Location'] = le.fit_transform(merged.Internship_Location)

#merged.to_csv('check.csv')

#Removing unwanted features and cleaning earliest start date
merged.Earliest_Start_Date=pd.to_datetime(merged['Earliest_Start_Date'])
merged.Start_Date = pd.to_datetime(merged['Start_Date'])
merged.Time_Difference = abs(merged.Earliest_Start_Date-merged.Start_Date)

#Calculating Time_Difference feature between earliest start date and Internship start date
Diff = [merged.Time_Difference[num].days for num in range(len(merged))]
merged['Time_Difference'] = Diff 

#Some final tweaks, removed StudentID,InternshipID, and a few unwanted columns
#Looks like I have all my features ready to train the algorithm. Phew! Finally!

merged = merged.drop(['Earliest_Start_Date','Start_Date','Student_ID','Internship_ID','Stream','Year_of_graduation','Profile','Start Date','End Date','Internship_Profile',
'Skills_required','Internship_category','Stipend_Type','Stipend1','Stipend2','Internship_deadline'],axis=1)

X = merged    
X_train = X[0:192582]
X_test = X[192582:len(X)]


X_new_train = X_train.ix[0:,[2,5,6,7,9,10,11,12,291,292]]
X_new_train = pd.concat([X_new_train,X_train.ix[0:,19:291]],axis=1)

X_new_test =  X_test.ix[0:,[2,5,6,7,9,10,11,12,291,292]]
X_new_test = pd.concat([X_new_test,X_test.ix[0:,19:291]],axis=1)
"""
Cross Validation
Recursive Feature Elimination
"""

decision_tree = tree.DecisionTreeClassifier()
decision_tree.fit(X_new_train,y_train)
scores = cross_validation.cross_val_score(decision_tree,X_new_train,y_train)
print(scores.mean())
predictions = decision_tree.predict(X_new_test)

rf = RandomForestClassifier(n_estimators=20, max_depth=None,min_samples_split=1, random_state=0)
scores = cross_validation.cross_val_score(rf, X_train, y_train)
print (scores.mean())
rf.fit(X_train,y_train)
predictions_rf = rf.predict(X_test)

extra_trees = ExtraTreesClassifier(n_estimators=10, max_depth=None,min_samples_split=1, random_state=0)
scores = cross_validation.cross_val_score(extra_trees, X_train, y_train)
print (scores.mean())
extra_trees.fit(X_new_train,y_train)
predictions_et_new = extra_trees.predict(X_new_test)
#print (decision_tree.feature_importances_)

gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=3, random_state=0)
gbc.fit(X_train,y_train)
gbc_predictions = gbc.predict(X_test)

abc = AdaBoostClassifier(n_estimators=100)
abc.fit(X_new_train,y_train)
predictions_abc = abc.predict(X_new_test)

scores = cross_validation.cross_val_score(decision_tree, X_train, y_train)
print (scores)

ensemble = BaggingClassifier(base_estimator=decision_tree,
                                max_features=1.0,
                                n_jobs=-1,
                                verbose = 1,                             
                                random_state=0)
ensemble.fit(X_train, y_train)
ensem_predictions = ensemble.predict(X_test)
scores = cross_validation.cross_val_score(ensemble, X_train, y_train)
print (scores)


# Getting the final submission file ready
final_submission = pd.DataFrame(test_raw.Internship_ID)
final_submission['Student_ID']=final_submission.Student_ID = test_raw.Student_ID
final_submission['Is_Shortlisted']=final_submission.Is_Shortlisted=predictions_abc
final_submission.to_csv('reduced_abc.csv')
  



