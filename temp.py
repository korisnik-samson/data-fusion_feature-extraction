import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer

# Load the F1 pitstop dataset (assuming it's already cleaned)
df = pd.read_csv('f1_pitstop_cleaned.csv')

# Display basic info about the dataset
print(df.info())
print(df.describe())

# Assuming 'duration' is the target variable (pit stop time in seconds)
# Adjust as needed based on your actual dataset
X = df.drop(['duration'], axis=1)
y = df['duration']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Identify column types
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

# Preprocessing pipeline
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

# Feature selection and PCA pipeline
def create_pipeline(model):
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('feature_selection', SelectKBest(score_func=f_regression, k=10)),
        ('pca', PCA(n_components=0.95)),  # Keep 95% of variance
        ('model', model)
    ])

# Define models and their hyperparameters for tuning
models = {
    'Linear Regression': {
        'model': LinearRegression(),
        'params': {}
    },
    'Ridge Regression': {
        'model': Ridge(),
        'params': {'model__alpha': [0.01, 0.1, 1.0, 10.0]}
    },
    'Random Forest': {
        'model': RandomForestRegressor(random_state=42),
        'params': {
            'model__n_estimators': [50, 100],
            'model__max_depth': [None, 10, 20],
            'feature_selection__k': [5, 10, 15]
        }
    },
    'Gradient Boosting': {
        'model': GradientBoostingRegressor(random_state=42),
        'params': {
            'model__n_estimators': [50, 100],
            'model__learning_rate': [0.01, 0.1],
            'feature_selection__k': [5, 10, 15]
        }
    }
}

# Train and evaluate models
results = {}
for name, config in models.items():
    print(f"\n{'-'*50}\nTraining {name}...\n{'-'*50}")

    # Create pipeline with the model
    pipeline = create_pipeline(config['model'])

    # Grid search for hyperparameter tuning (if params exist)
    if config['params']:
        search = GridSearchCV(pipeline, config['params'], cv=5, scoring='neg_mean_squared_error')
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        print(f"Best parameters: {search.best_params_}")
    else:
        best_model = pipeline
        best_model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Store results
    results[name] = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'model': best_model
    }

    # Visualizations
    plt.figure(figsize=(12, 5))

    # Predicted vs Actual plot
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    plt.title(f'{name}: Predicted vs Actual')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')

    # Residual plot
    plt.subplot(1, 2, 2)
    residuals = y_test - y_pred
    sns.histplot(residuals, kde=True)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.title(f'{name}: Residuals')
    plt.xlabel('Residual')
    plt.ylabel('Count')

    plt.tight_layout()
    plt.show()

# Compare model performance
results_df = pd.DataFrame({name: {'RMSE': res['RMSE'], 'R2': res['R2'], 'MAE': res['MAE']}
                           for name, res in results.items()}).T
print("\nModel Performance Comparison:")
print(results_df)

# Plot model comparison
plt.figure(figsize=(12, 6))
sns.barplot(x=results_df.index, y='RMSE', data=results_df)
plt.title('RMSE Comparison')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Feature importance for the best model (if available)
best_model_name = results_df['RMSE'].idxmin()
print(f"\nBest model: {best_model_name}")
best_pipeline = results[best_model_name]['model']

# For Random Forest and Gradient Boosting, we can extract feature importance
if 'Random Forest' in best_model_name or 'Gradient Boosting' in best_model_name:
    try:
        # Get feature names (may require backward transformation through pipeline)
        preprocessed_X_train = best_pipeline.named_steps['preprocessor'].transform(X_train)
        feature_names = best_pipeline.named_steps['preprocessor'].get_feature_names_out()

        # Get selected features
        if 'feature_selection' in best_pipeline.named_steps:
            selected_indices = best_pipeline.named_steps['feature_selection'].get_support()
            selected_features = feature_names[selected_indices]
        else:
            selected_features = feature_names

        # Get feature importances from the model
        importances = best_pipeline.named_steps['model'].feature_importances_

        # Create DataFrame of features and their importances
        feature_importance_df = pd.DataFrame({
            'Feature': selected_features,
            'Importance': importances
        }).sort_values('Importance', ascending=False)

        # Plot feature importances
        plt.figure(figsize=(10, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(15))
        plt.title(f'Top 15 Features - {best_model_name}')
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Couldn't extract feature importance: {e}")