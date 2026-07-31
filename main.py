import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


df=pd.read_csv("Training.csv")

#print(df.isnull().sum())
df=df.drop(['Unnamed: 133'],axis=1)
#print(df.columns)

le=LabelEncoder()
encoded_labels=le.fit_transform(df['prognosis'])
print(encoded_labels)

X=df.drop(['prognosis'],axis=1)
y=encoded_labels

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
#scaler=StandardScaler()
#X_scaled_train=scaler.fit_transform(X_train)
#X_scaled_test=scaler.fit(X_test)
rf=RandomForestClassifier(random_state=42,n_estimators=100)
rf.fit(X_train,y_train)
#pred=rf.predict(X_test)
#print(pred)
#print(le.classes_)
#getting user input & convert it into array
u_input=input("Enter the symptoms u have : " )
u_input=u_input.split(",")


zero_array=np.zeros(132).reshape(1,132)
#print(zero_array.shape)
list_diseases=list(df.columns.drop(['prognosis']))
#print(list_disease)

#for changing 0 to 1 for using it for prediction  
for symptoms in u_input:
    idx=list_diseases.index(symptoms.strip())
    zero_array[0][idx]=1

print(zero_array)

#getting prediction

pred=rf.predict(zero_array)

predicted_disese=le.inverse_transform(pred)
print(predicted_disese)