import numpy as np 
import pandas as pd 

dataset = pd.read_csv('dataset/titanic/train.csv')


dataset.drop(['Ticket', 'Cabin'], axis=1,inplace=True)


dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)



dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

dataset['Title'] = dataset['Title'].map(title_mapping)
dataset['Title'] = dataset['Title'].fillna(0)

dataset = dataset.drop(['Name', 'PassengerId'], axis=1)

dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

guess_ages = np.zeros((2,3))

print('null ',dataset.isnull().sum())
print(dataset.head())
for i in range(0, 2):
    for j in range(0, 3):
        guess_df = dataset[(dataset['Sex'] == i) & \
                                (dataset['Pclass'] == j+1)]['Age'].dropna()

        # age_mean = guess_df.mean()
        # age_std = guess_df.std()
        # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

        age_guess = guess_df.median()
        print(age_guess)

        # Convert random age float to nearest .5 age
        guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
            
for i in range(0, 2):
    for j in range(0, 3):
        dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\
                'Age'] = guess_ages[i,j]

dataset['Age'] = dataset['Age'].astype(int)

dataset.loc[ dataset['Age'] <= 12, 'Age'] = 0
dataset.loc[(dataset['Age'] > 12) & (dataset['Age'] <= 23), 'Age'] = 1
dataset.loc[(dataset['Age'] > 23) & (dataset['Age'] <= 34), 'Age'] = 2
dataset.loc[(dataset['Age'] > 34) & (dataset['Age'] <= 46), 'Age'] = 3
dataset.loc[(dataset['Age'] > 46) & (dataset['Age'] <= 57), 'Age'] = 4
dataset.loc[(dataset['Age'] > 57) & (dataset['Age'] <= 68), 'Age'] = 5
dataset.loc[ dataset['Age'] > 68, 'Age'] = 6

dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

dataset['IsAlone'] = 0
dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

dataset = dataset.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)

dataset['Age*Class'] = dataset.Age * dataset.Pclass

freq_port = dataset.Embarked.dropna().mode()[0]
dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

dataset['Fare'].fillna(dataset['Fare'].dropna().median(), inplace=True)

dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
dataset['Fare'] = dataset['Fare'].astype(int)

survived = dataset['Survived']

dataset = dataset.drop('Survived',axis=1)
dataset['Survived']=survived
print(dataset.columns)
dataset.to_csv('dataset/df.csv')


