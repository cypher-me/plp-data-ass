import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib
matplotlib.use('Agg') 

print("Task 1: Load and Explore the Dataset")

url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
df = pd.read_csv(url)

print("\nFirst 5 rows of the dataset:")
print(df.head())

print("\nDataset information:")
print(df.info())

print("\nMissing values in each column:")
print(df.isnull().sum())

df_with_missing = df.copy()
df_with_missing.loc[10:15, 'sepal_length'] = np.nan
df_with_missing.loc[20:25, 'petal_width'] = np.nan

print("\nDataset with artificially created missing values:")
print(df_with_missing.isnull().sum())

df_cleaned = df_with_missing.copy()
df_cleaned['sepal_length'] = df_cleaned['sepal_length'].fillna(df_cleaned['sepal_length'].mean())
df_cleaned['petal_width'] = df_cleaned['petal_width'].fillna(df_cleaned['petal_width'].mean())

print("\nDataset after cleaning (filling missing values):")
print(df_cleaned.isnull().sum())

print("\nTask 2: Basic Data Analysis")

print("\nBasic statistics of numerical columns:")
print(df.describe())

print("\nMean of numerical columns grouped by species:")
print(df.groupby('species').mean())

print("\nInteresting findings:")
print("1. Setosa has the smallest petal length and width.")
print("2. Virginica has the largest petal length and width.")
print("3. There's a clear separation between species based on petal characteristics.")

print("\nTask 3: Data Visualization")

sns.set_theme(style="whitegrid")

plt.figure(figsize=(14, 10))

np.random.seed(42)
dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
time_series_data = pd.DataFrame({
    'date': dates,
    'setosa_count': np.random.randint(10, 30, 50),
    'versicolor_count': np.random.randint(15, 35, 50),
    'virginica_count': np.random.randint(20, 40, 50)
})

plt.subplot(2, 2, 1)
plt.plot(time_series_data['date'], time_series_data['setosa_count'], label='Setosa')
plt.plot(time_series_data['date'], time_series_data['versicolor_count'], label='Versicolor')
plt.plot(time_series_data['date'], time_series_data['virginica_count'], label='Virginica')
plt.title('Daily Count of Iris Species (Sample Data)')
plt.xlabel('Date')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()

plt.subplot(2, 2, 2)
species_means = df.groupby('species').mean().reset_index()
x = species_means['species']
y = species_means['petal_length']
plt.bar(x, y, color=['blue', 'green', 'red'])
plt.title('Average Petal Length by Species')
plt.xlabel('Species')
plt.ylabel('Average Petal Length (cm)')
plt.tight_layout()

plt.subplot(2, 2, 3)
plt.hist(df['sepal_length'], bins=10, color='skyblue', edgecolor='black')
plt.title('Distribution of Sepal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Frequency')
plt.tight_layout()

plt.subplot(2, 2, 4)
for species in df['species'].unique():
    subset = df[df['species'] == species]
    plt.scatter(subset['sepal_length'], subset['petal_length'], label=species, alpha=0.7)
plt.title('Sepal Length vs Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend()
plt.tight_layout()


plt.savefig('iris_data_analysis.png')
print("\nMain visualizations saved as 'iris_data_analysis.png'")
plt.close()

sns.pairplot(df, hue='species')
plt.suptitle('Pair Plot of Iris Dataset', y=1.02)
plt.savefig('iris_pairplot.png')
print("Pair plot saved as 'iris_pairplot.png'")
plt.close()

print("\nData analysis and visualization tasks completed successfully!")