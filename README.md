🍷 Wine Quality Prediction Using Machine Learning

An end-to-end Machine Learning project that predicts whether a wine is Good Quality or Bad Quality based on its physicochemical properties.

This project covers complete ML workflow:

Data Cleaning
Exploratory Data Analysis (EDA)
Handling Multicollinearity
Feature Scaling
PCA (Dimensionality Reduction)
Class Imbalance Handling (SMOTE)
Model Training & Comparison
GUI Deployment (Tkinter)
📌 Problem Statement

Wine quality is influenced by multiple chemical properties such as:

Fixed Acidity
Volatile Acidity
Citric Acid
Residual Sugar
Chlorides
Free Sulfur Dioxide
Total Sulfur Dioxide
Density
pH
Sulphates
Alcohol

The goal is to predict whether a wine is:

1 → Good Quality Wine
0 → Bad Quality Wine
🚀 Step-by-Step Implementation
1️⃣ Data Loading & Exploration
Loaded dataset using pandas
Checked:
Missing values
Data types
Statistical summary
Visualized distribution of target variable
df.info()
df.describe()
df['quality'].value_counts()
2️⃣ Exploratory Data Analysis (EDA)
🔥 Correlation Heatmap

Created correlation matrix to detect:

Strongly correlated features
Multicollinearity
Feature redundancy
import seaborn as sns
sns.heatmap(df.corr(), annot=True)
📌 Why Heatmap Was Important

It helped identify:

Free sulfur dioxide ↔ Total sulfur dioxide (high correlation)
Density ↔ Alcohol
Residual sugar ↔ Density

Multicollinearity can:

Distort linear model coefficients
Increase variance
Reduce interpretability
3️⃣ Handling Target Variable

Converted wine quality into binary classification:

df['quality'] = df['quality'].apply(lambda x: 1 if x >= 7 else 0)
4️⃣ Feature & Target Split
X = df.drop('quality', axis=1)
y = df['quality']
5️⃣ Handling Class Imbalance

The dataset was imbalanced (more bad wines than good wines).

Applied SMOTE (Synthetic Minority Over-sampling Technique):

from imblearn.over_sampling import SMOTE
X_res, y_res = SMOTE().fit_resample(X, y)
📌 Why SMOTE?

Without balancing:

Model showed misleading 99% accuracy
Low recall for good wine detection

After SMOTE:

Precision improved
Recall improved
F1 score improved significantly
6️⃣ Feature Scaling

Standardized features before PCA and model training:

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_res)
7️⃣ PCA (Principal Component Analysis)

Applied PCA to:

Reduce multicollinearity
Remove redundancy
Improve generalization
Retain maximum information
from sklearn.decomposition import PCA
pca = PCA(n_components=0.90)
X_pca = pca.fit_transform(X_scaled)
📌 PCA Result

Retained ~90% of total variance
Reduced dimensionality while preserving information.

8️⃣ Train-Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y_res, test_size=0.2, random_state=42
)
9️⃣ Model Training

Trained multiple models:

Logistic Regression
SVC
KNN
Decision Tree
Random Forest
Gradient Boosting

Example:

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
🔟 Model Evaluation

Evaluated using:

Accuracy
Precision
Recall
F1 Score
Confusion Matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
🏆 Best Model

Random Forest

Accuracy ≈ 91.86%
Strong balance of precision & recall
1️⃣1️⃣ Model Saving

Saved trained model using joblib:

import joblib
joblib.dump(rf, "wine_quality_prediction")
1️⃣2️⃣ GUI Deployment (Tkinter)

Built a simple GUI where users can:

Enter chemical properties
Click Predict
Get result: Good / Bad Quality Wine

Prediction pipeline inside GUI:

new_data = pd.DataFrame({...})

test = pca.transform(scaler.transform(new_data))
prediction = model.predict(test)
📊 Model Performance Comparison
Model	Accuracy (%)
Logistic Regression	81.56
SVC	87.88
KNN	87.70
Decision Tree	86.98
Random Forest	91.86
Gradient Boosting	87.88
📈 Key Learnings

✔ Accuracy can be misleading in imbalanced datasets
✔ Heatmaps help detect multicollinearity
✔ PCA reduces redundancy while preserving information
✔ SMOTE improves minority class detection
✔ Ensemble models perform better for complex patterns
✔ Evaluation metrics beyond accuracy are crucial

🛠️ Tech Stack
Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-Learn
Imbalanced-Learn
Tkinter
Joblib
💡 Future Improvements
Hyperparameter tuning (GridSearchCV)
Deploy using Flask / Streamlit
Add feature importance visualization
Try XGBoost / LightGBM
Deploy to cloud
👨‍🔬 Author

Aayush Gautam
Data Scientist | ML Enthusiast | Fitness Coach
Building models that mix science with storytelling 🍷🤖
