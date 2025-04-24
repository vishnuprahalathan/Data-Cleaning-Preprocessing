# Data-Cleaning-Preprocessing


This project is a part of an AI & ML internship assignment, focused on preparing raw data for machine learning using the Titanic dataset.

Objective
To clean and preprocess the Titanic dataset by handling missing data, encoding categorical variables, standardizing features, and removing outliers—then saving the cleaned dataset to a designated folder.

Tools & Libraries
- Python
- Pandas
- NumPy
- Seaborn
- Matplotlib
- Scikit-learn

Steps Performed

1. Dataset Loading
   - Loaded `Titanic-Dataset.csv` using `pandas`.

2. Data Exploration
   - Displayed basic info, descriptive statistics, and checked for missing values.

3. Missing Value Treatment
   - Filled missing `Age` values with median.
   - Filled missing `Embarked` values withmode.
   - Dropped the `Cabin` column due to excessive missing values.

4. Encoding Categorical Features
   - Applied **Label Encoding** to the `Sex` column.
   - Applied **One-Hot Encoding** to the `Embarked` column with `drop_first=True` to avoid dummy variable trap.

5. Feature Scaling
   - Standardized `Age` and `Fare` using **StandardScaler** for zero mean and unit variance.

6. Outlier Detection and Removal
   - Used boxplot* for visualization.
   - Removed outliers from the `Fare` column using the IQR method.

7. Saving the Cleaned Data
   - Saved the cleaned dataset to:
     ```
     C:\Users\Vishnu Prahalathan\Desktop\SCREENSHOTS\titanic_cleaned.csv
     ```

Outcome
A ready-to-use, cleaned dataset suitable for training ML models has been created and saved.

#Dataset Source
[Kaggle: Titanic - Machine Learning from Disaster](https://www.kaggle.com/datasets/yasserh/titanic-dataset)


