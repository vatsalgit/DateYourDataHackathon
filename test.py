# -*- coding: utf-8 -*-
"""
----------Working with testing data to calculate test features----------

"""
test_raw = pd.read_csv('test-date-your-data/test.csv')
 
test_raw = pd.merge(test_raw,student, on='Student_ID',how='left')
test_raw = pd.merge(test_raw,internship,on='Internship_ID',how='left')

test_raw.Percentage_PG = test_raw.Performance_PG/test_raw.PG_scale

test_raw.Percentage_UG = test_raw.Performance_UG/test_raw.UG_Scale

test_raw['Percentage_PG']=test_raw.Percentage_PG
test_raw['Percentage_UG']=test_raw.Percentage_UG

test_raw = test_raw.drop(test_raw.columns[[14,15,16,17,18]], axis=1) 
                 
#Converting categorical variables to numeric attributes 
test_raw = test_raw.fillna('NA')         
le = preprocessing.LabelEncoder()
test_raw['Expected_Stipend'] = le.fit_transform(test_raw.Expected_Stipend)
test_raw['Preferred_location'] = le.fit_transform(test_raw.Preferred_location)
test_raw['Institute_Category'] = le.fit_transform(test_raw.Institute_Category)
test_raw['Institute_location'] = le.fit_transform(test_raw.Institute_location)
test_raw['hometown'] = le.fit_transform(test_raw.hometown)
test_raw['Degree'] = le.fit_transform(test_raw.Degree)
test_raw['Current_year'] = le.fit_transform(test_raw.Current_year)
test_raw['Experience_Type'] = le.fit_transform(test_raw.Experience_Type)
test_raw['Location'] = le.fit_transform(test_raw.Location)
test_raw['Internship_Type'] = le.fit_transform(test_raw.Internship_Type)
test_raw['Internship_Location'] = le.fit_transform(test_raw.Internship_Location)

test_raw.Earliest_Start_Date=pd.to_datetime(test_raw['Earliest_Start_Date'])
test_raw.Start_Date = pd.to_datetime(test_raw['Start_Date'])
test_raw.Time_Difference = abs(test_raw.Earliest_Start_Date-test_raw.Start_Date)

Diff_Test = [test_raw.Time_Difference[num].days for num in range(len(test_raw))]
test_raw['Time_Difference'] = Diff_Test 

