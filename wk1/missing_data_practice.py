import pandas as pd

df = pd.read_csv('titanic_train.csv')

print(df.isna().sum())
print(df['Age'].describe()) # Establishes a base description of 'Age' before filling in missing values

# Output:
# Survived         0
# Pclass           0
# Name             0
# Sex              0
# Age            177
# SibSp            0
# Parch            0
# Ticket           0
# Fare             0
# Cabin          687
# Embarked         2
# count    714.000000
# mean      29.699118
# std       14.526497
# min        0.420000
# 25%       20.125000
# 50%       28.000000
# 75%       38.000000
# max       80.000000

# 177 datapoints missing from Age; 687 from Cabin; 2 from Embarked

df['Embarked']= df['Embarked'].fillna(df['Embarked'].mode()[0]) # Fill missing values in 'Embarked' column with the mode (most frequent value)
# Test to make sure that the missing values have been filled in
print(df.isna().sum())

# Age has many more missing values and may be dependent on other variables, so we will inspect dependency by grouping
df.groupby(['Pclass', 'Sex'])['Age'].median()
# Fill missing values in 'Age' column with the median age of the group (Pclass and Sex); better than filling with overall median
# .transform('median') applies the median function to each group and fills in the missing values
df['Age'] = df['Age'].fillna(df.groupby(['Pclass', 'Sex'])['Age'].transform('median'))
# Test to make sure that the missing values have been filled in
print(df.isna().sum())
print(df['Age'].describe()) # Check that the age descriptions is still similar (sanity check)
# Output: reasonable; similar mean and median, std does not collapse
# count    891.000000
# mean      29.112424
# std       13.304424
# min        0.420000
# 25%       21.500000
# 50%       26.000000
# 75%       36.000000
# max       80.000000

# Cabin is missing a large majority of its values, so we drop the column and create a column with missing values
df['Cabin_Missing'] = df['Cabin'].isna().astype(int) # Create new column with 1 if missing, 0 if not missing
df = df.drop(columns=['Cabin'])
print(df.groupby('Cabin_Missing')['Survived'].mean()) # Check if 1 in Cabin_Missing is associated with survival
# Output:
# Cabin_Missing
# 0    0.666667 --> 67% of those with a cabin survived
# 1    0.299854 --> 30% of those without a cabin survived
# Test to make sure that the column is dropped and the new column is created without missing values
print(df.isna().sum())

# Now, all missing values are handled. We can proceed with cleaning the names to be more useful
df['Title'] = (df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)) # Extract titles from the 'Name' column, adds a new column 'Title'
df['Title'] = df['Title'].replace(['Lady','Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'],'Rare') # Replace rare titles with 'Rare' to avoid overfitting
# We can use this new 'Title' column to predict survival based on the title of the passenger, or examine correlation with other variables (Pclass and Age, probably)