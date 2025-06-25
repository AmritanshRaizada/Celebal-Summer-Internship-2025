import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load Titanic dataset
df = sns.load_dataset('titanic')  # Or replace with pd.read_csv('titanic.csv') if using local file

# Clean data
df['age'] = df['age'].fillna(df['age'].median())
df.dropna(subset=['embarked'], inplace=True)
if 'deck' in df.columns:
    df.drop(columns=['deck'], inplace=True)

# Plot 1: Age Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df['age'], bins=30, kde=True)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.savefig('age_distribution.png')
plt.close()

# Plot 2: Survival by Gender
plt.figure(figsize=(8, 5))
sns.countplot(x='survived', hue='sex', data=df)
plt.title('Survival by Gender')
plt.xlabel('Survived')
plt.ylabel('Count')
plt.savefig('survival_by_gender.png')
plt.close()

# Plot 3: Fare by Survival
plt.figure(figsize=(8, 5))
sns.boxplot(x='survived', y='fare', data=df)
plt.title('Fare by Survival')
plt.xlabel('Survived')
plt.ylabel('Fare')
plt.savefig('fare_by_survival.png')
plt.close()

# Plot 4: Fare by Embarkation Point
plt.figure(figsize=(8, 5))
sns.boxplot(x='embarked', y='fare', data=df)
plt.title('Fare by Embarkation Point')
plt.xlabel('Embarked')
plt.ylabel('Fare')
plt.savefig('fare_by_embarked.png')
plt.close()

print("✅ EDA Complete. All 4 plots saved.")
