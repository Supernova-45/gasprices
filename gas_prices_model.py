import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
import shap
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('gas_data.csv')

# Prepare feature and target variables
features = ['Real Price of Gasoline', 'Annual Vehicle Miles', 'Vehicle Fuel Efficiency', 'Real U.S. GDP', 'Threat of Climate Change', 'Unemployment %']
targets = ['Income Quintile 1', 'Income Quintile 2', 'Income Quintile 3', 'Income Quintile 4', 'Income Quintile 5']

# Fill missing values in 'Threat of Climate Change**' column using mean imputation
data['Threat of Climate Change'].fillna(data['Threat of Climate Change'].mean(), inplace=True)

# Verify there are no other missing values in the dataset
if data.isnull().sum().any():
    print("Warning: There are still missing values in the dataset.")
    # Handling remaining missing values using SimpleImputer
    imputer = SimpleImputer(strategy='mean')
    data[features] = imputer.fit_transform(data[features])


# Split the data into training and testing sets (random sample of 10 years for testing)
test_years = np.random.choice(data['Year'].unique(), size=10, replace=False)
train_data = data[~data['Year'].isin(test_years)]
test_data = data[data['Year'].isin(test_years)]

# Initialize a dictionary to store results
results = {}

# Subplots
fig, axes = plt.subplots(5, 1, figsize=(10, 30))
fig.tight_layout(pad=5.0)

for i, target in enumerate(targets):
    X_train = train_data[features]
    y_train = train_data[target]
    X_test = test_data[features]
    y_test = test_data[target]
    
    # Initialize and train the model
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Calculate accuracy (Mean Squared Error in this case)
    mse = mean_squared_error(y_test, predictions)
    results[target] = mse
    
    # SHAP values
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)
    
    # Plot SHAP values# Plot SHAP values
    shap.summary_plot(shap_values, X_test, feature_names=features, show=False, plot_type='bar', ax=axes[i])
    axes[i].set_title(f'SHAP Values for {target}')
# Display results
for target, mse in results.items():
    print(f'Mean Squared Error for {target}: {mse}')

plt.savefig(f'shap_values.png')
plt.show()