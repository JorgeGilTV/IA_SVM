import numpy as np
import pandas as pd

import os
for dirname, _, filenames in os.walk('input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train_data = pd.read_csv("train.csv")
print(train_data.head())

test_data = pd.read_csv("test.csv")
print(test_data.head())

women = train_data.loc[train_data.Sex == "female", :]
#rate_women = sum(women)/len(women)
print(women.info)

men = train_data.loc[train_data.Sex == "male", :]
print(men.info)

from sklearn.ensemble import RandomForestClassifier
y = train_data["Survived"]

features = ["Pclass","Sex","SibSp","Parch"]
X= pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X,y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived':predictions})
output.to_csv('submission.csv',index=False)
print("Your submission was successfully saved")
