# IDX-Exchange-Housing-Prices-Project
This project builds machine learning models to predict real estate closing prices using MLS (Multiple Listing Service) data. The dataset spans multiple monthly CSV files and includes residential property listings, with a focus on Single Family Residences.
The goal is to analyze key housing features and evaluate how well different regression models can predict the closing price of a given property. 

**The dataset(Files Used):**

CRMLSSold202502.csv,
CRMLSSold202503.csv,
CRMLSSold202504.csv,
CRMLSSold202505.csv,
CRMLSSold202506.csv,
CRMLSSold202507.csv,
CRMLSSold202508.csv,
CRMLSSold202509.csv,
CRMLSSold202510.csv,
CRMLSSold202511.csv,
CRMLSSold202512.csv,


**Properties:**
PropertyType = Residential
PropertySubType = SingleFamilyResidence

**Key Features:**
ClosePrice (target variable)
LivingArea
BedroomsTotal
BathroomsTotalInteger
LotSizeSquareFeet
PropertyType
PropertySubType

**Data Preprocessing:**
1. Data Cleaning
Removed duplicates
Dropped columns with excessive missing values (>35%)
Dropped irrelevant or high-cardinality identifier fields (e.g., ListingId, agent info, addresses)
2. Missing Values
Rows/columns with significant null values were removed
Final dataset was cleaned using dropna()
3. Encoding
Categorical variables were converted using Label Encoding
4. Feature Scaling
Applied:
MinMaxScaler
StandardScaler
(used for normalization experiments)
5. Feature Engineering
Created additional feature:
bed_bath_ratio = BedroomsTotal / (BathroomsTotalInteger + 0.5)

**Models Tested**
1. Linear Regression
Baseline model for performance comparison
2. Random Forest Regressor
100 decision trees
Captures nonlinear relationships
Feature importance analysis performed
3. XGBoost Regressor
Gradient boosting model
Tuned with:
n_estimators = 200
max_depth = 3
learning_rate = 0.1
subsample = 0.8

**Model Evaluation Metrics**:

Models were evaluated using:
R² Score
Mean Squared Error (MSE)
Root Mean Squared Error (RMSE)
Median Absolute Percentage Error (mdAPE)

Linear Regression:
Used as baseline model
Performed worse than tree-based models due to nonlinear relationships in housing data

Random Forest (Best Performing Model):
R² Score: ~0.863
mdAPE: ~8.69%
Top Features:
LivingArea (19.03%)
BathroomsTotalInteger (19.03%)
PropertyType (15.21%)

XGBoost Model:
R² Score: ~0.796
mdAPE: ~18.75%
Top Features:
PropertyType (19.44%)
BathroomsTotalInteger (14.14%)
LivingArea (13.53%)


📌 Key Insights
Living Area and Bathrooms are the strongest predictors of house price.
Tree-based models significantly outperform linear regression.
Feature engineering (like bed-bath ratio) improved interpretability and how the features had an impact.
Housing prices are highly nonlinear, making ensemble models more suitable.
