# ML Learning Journey 🚀

This repository contains my machine learning learning journey and will be continuously updated as I learn and implement new concepts.

## 📚 Learning Topics Covered

### 📥 Data Gathering

Collecting data from various sources is the first step in any data science project. This section covers different methods for acquiring data.

- [**From API**](data_gathering/from_api.ipynb) - Fetching data from REST APIs
  - **Theory**: API requests, JSON parsing, data extraction, handling pagination
  - **Use Cases**: Real-time data collection, third-party data integration

- [**Web Scraping**](data_gathering/web_scrapping.ipynb) - Extracting data from websites
  - **Theory**: HTML parsing, XPath/CSS selectors, handling dynamic content
  - **Tools**: BeautifulSoup, requests, handling anti-scraping measures

- [**From CSV**](data_gathering/with_csv.ipynb) - Working with CSV files
  - **Theory**: File I/O, data loading, handling different encodings
  - **Use Cases**: Local data storage, data export/import workflows

### 🔧 Data Preprocessing

#### Feature Scaling & Normalization
Data preprocessing is a crucial step in machine learning that transforms raw data into a format suitable for ML algorithms. Feature scaling ensures all features contribute equally to the model by bringing them to similar scales.

- [**Numerical Data Preprocessing**](data_preprocessing/numerical_ds_preprocessing.ipynb) - StandardScaler and train-test split
  - **Theory**: StandardScaler transforms features to have mean=0 and std=1, essential for algorithms sensitive to feature scales
  - **Use Cases**: SVM, Neural Networks, PCA, KMeans clustering
  
- [**Data Standardization & Train-Test Split**](data_preprocessing/d_standaize_train_test_split.ipynb) - Feature scaling and data splitting
  - **Theory**: Proper train-test splitting prevents data leakage and ensures unbiased model evaluation
  - **Best Practices**: 80-20 or 70-30 splits, stratified sampling for classification

#### Categorical Data Processing  
Categorical variables need special handling since ML algorithms work with numerical data. Different encoding methods have different implications for model performance.

- [**Label Encoding**](data_preprocessing/label_encoding.ipynb) - Converting categorical data to numerical
  - **Theory**: Maps categories to integers, preserves ordinal relationships but may introduce artificial ordering
  - **When to Use**: Ordinal data, tree-based algorithms
  - **Alternatives**: One-hot encoding for nominal data, target encoding for high cardinality

#### Missing Data Handling
Missing data is common in real-world datasets and requires careful treatment to avoid biased or unreliable models.

- [**Handle Missing Values**](data_preprocessing/handle_missing_values.ipynb) - Techniques for dealing with missing data
  - **Theory**: Missing Completely at Random (MCAR), Missing at Random (MAR), Missing Not at Random (MNAR)
  - **Techniques**: Deletion (listwise/pairwise), Imputation (mean/median/mode), Advanced methods (KNN, iterative)
  - **Impact**: Different methods affect model performance and interpretability differently

#### Class Imbalance Solutions
Imbalanced datasets can lead to biased models that perform poorly on minority classes, especially in classification problems.

- [**Handle Imbalanced Data**](data_preprocessing/handle_imbalanced_dp.ipynb) - Methods for handling imbalanced datasets
  - **Theory**: Accuracy paradox, precision-recall tradeoff, cost-sensitive learning
  - **Techniques**: Oversampling (SMOTE), Undersampling, Class weights, Ensemble methods
  - **Evaluation**: F1-score, AUC-ROC, Precision-Recall curves instead of accuracy

#### Natural Language Processing (NLP)
Text data requires specialized preprocessing to extract meaningful features that ML algorithms can understand.

- [**Text Data Preprocessing**](data_preprocessing/text_ds_preprocessing.ipynb) - Text cleaning and feature extraction
  - **Theory**: Tokenization, stopword removal, stemming/lemmatization, TF-IDF, Bag of Words
  - **Challenges**: Noise removal, handling different languages, maintaining semantic meaning
  
- [**Text Data Preprocessing 2**](data_preprocessing/text_ds_preprocessing2.ipynb) - Advanced text processing techniques
  - **Theory**: N-grams, Word embeddings (Word2Vec, GloVe), Named Entity Recognition (NER)
  - **Modern Approaches**: BERT embeddings, transformer-based preprocessing

### 🤖 Machine Learning Projects

#### Supervised Learning - Classification
Classification problems predict discrete categories or classes. These projects demonstrate various classification algorithms and their applications.

- [**Diabetes Prediction**](projects/diabeties_prediction.ipynb) - Classification model for diabetes prediction
  - **Theory**: Binary classification, logistic regression, decision boundaries, ROC curves
  - **Domain**: Healthcare analytics, risk assessment, feature importance in medical diagnosis
  - **Algorithms**: Logistic Regression, Random Forest, SVM

- [**Sleep Disorder Prediction**](projects/Sleep%20Disorder%20Prediction.ipynb) - Classification model for sleep disorder analysis
  - **Theory**: Multi-class classification, health indicators analysis, lifestyle factor impact
  - **Domain**: Healthcare analytics, sleep medicine, wellness prediction
  - **Features**: Sleep duration, quality metrics, lifestyle factors

- [**Fake News Detection**](projects/fake_news_prediction.ipynb) - NLP classification for fake news detection
  - **Theory**: Text classification, feature extraction from text, handling high-dimensional data
  - **Domain**: Natural Language Processing, information verification, social media analysis
  - **Challenges**: Bias detection, semantic understanding, scalability

- [**Wine Quality Prediction**](projects/wine_quality_prediction.ipynb) - Multi-class classification for wine quality
  - **Theory**: Multi-class classification, ordinal vs nominal categories, class imbalance
  - **Domain**: Quality control, sensory data analysis, expert system modeling
  - **Evaluation**: Confusion matrices, macro/micro averaging, weighted metrics

- [**Sonar Rocks vs Mine Classification**](projects/sonar_rocks_vs_mine_predition.ipynb) - Binary classification using sonar data
  - **Theory**: Signal processing, pattern recognition, feature selection in high-dimensional data
  - **Domain**: Military applications, underwater object detection, sensor data analysis
  - **Algorithms**: Neural networks, ensemble methods, dimensionality reduction

- [**Loan Status Prediction**](projects/loan_status_prediction.ipynb) - Predicting loan approval status
  - **Theory**: Risk modeling, financial scoring systems, fairness in ML, regulatory compliance
  - **Domain**: Fintech, credit scoring, automated decision systems
  - **Considerations**: Bias mitigation, interpretability, business constraints

#### Supervised Learning - Regression
Regression problems predict continuous numerical values. These projects showcase different regression techniques and their real-world applications.

- [**House Price Prediction**](projects/house_price_prediction.ipynb) - Regression model for house prices
  - **Theory**: Linear regression, polynomial features, regularization (Ridge/Lasso), multicollinearity
  - **Domain**: Real estate valuation, economic modeling, location-based pricing
  - **Challenges**: Non-linear relationships, outlier handling, feature engineering

- [**House Price Prediction 2**](projects/house_price_prediction2.ipynb) - Advanced regression techniques for house prices
  - **Theory**: Advanced feature engineering, ensemble regression methods, cross-validation
  - **Domain**: Real estate valuation, comparative market analysis
  - **Improvements**: Hyperparameter tuning, model stacking, error analysis

- [**Car Price Prediction**](projects/price_card_prediction.ipynb) - Regression model for car prices
  - **Theory**: Depreciation modeling, categorical feature impact, interaction terms
  - **Domain**: Automotive industry, used car markets, asset valuation
  - **Features**: Brand reputation, mileage, age, market conditions

#### Time Series Prediction
Time series analysis involves predicting future values based on historical patterns, considering temporal dependencies and seasonality.

- [**Gold Price Prediction**](projects/gold_price_prediction.ipynb) - Time series prediction for gold prices
  - **Theory**: Time series decomposition, autocorrelation, stationarity, ARIMA models
  - **Domain**: Financial markets, commodity trading, economic indicators
  - **Challenges**: Market volatility, external factors, non-stationarity

#### Exploratory Data Analysis (EDA)
Understanding and visualizing data to uncover patterns, anomalies, and insights before building models.

- [**Pokemon Data Analysis**](projects/pokemon.ipynb) - Comprehensive EDA on Pokemon dataset
  - **Theory**: Data exploration, statistical analysis, visualization techniques
  - **Domain**: Gaming analytics, character statistics analysis
  - **Skills**: Data visualization, correlation analysis, feature distribution

#### Key Machine Learning Concepts Covered:
- **Model Selection**: Cross-validation, bias-variance tradeoff, overfitting prevention
- **Feature Engineering**: Domain knowledge incorporation, automated feature selection
- **Model Evaluation**: Appropriate metrics for different problem types, validation strategies
- **Hyperparameter Tuning**: Grid search, random search, Bayesian optimization
- **Ensemble Methods**: Bagging, boosting, voting classifiers for improved performance

## 📊 Datasets & Data Understanding

Understanding your data is the foundation of successful machine learning. Each dataset presents unique challenges and learning opportunities.

### Dataset Characteristics:
| Dataset | Type | Problem | Learning Focus |
|---------|------|---------|----------------|
| `cardekho.csv` | Tabular | Regression | Feature engineering, categorical variables, price prediction |
| `diabetes.csv` | Tabular | Binary Classification | Healthcare analysis, medical decision support |
| `gld_price_data.csv` | Time Series | Forecasting | Financial modeling, trend prediction |
| `loan_data.csv` | Tabular | Binary Classification | Risk modeling, fairness in AI |
| `sonar.all-data.csv` | Sensor | Binary Classification | Signal processing, pattern recognition |
| `Pokemon.csv` | Tabular | EDA/Classification | Gaming analytics, character analysis |
| `Sleep_health_and_lifestyle_dataset.csv` | Tabular | Classification | Health analytics, lifestyle impact |
| `profiles.json` | JSON | Various | JSON data handling, profile analysis |
| `train.csv` | Tabular | Various | Multi-purpose dataset |

### Data Science Workflow Concepts:
1. **Exploratory Data Analysis (EDA)**: Understanding data distributions, correlations, outliers
2. **Data Quality Assessment**: Missing values, duplicates, inconsistencies, data validation
3. **Feature Understanding**: Domain knowledge application, feature relationships, business context
4. **Target Variable Analysis**: Distribution, class balance, outliers, transformation needs

## 🛠️ Technologies & Libraries Used
- **Python** - Primary programming language
- **pandas** - Data manipulation and analysis
- **scikit-learn** - Machine learning algorithms and preprocessing
- **numpy** - Numerical computations
- **matplotlib/seaborn** - Data visualization
- **requests/BeautifulSoup** - Web scraping and API calls
- **nltk** - Natural language processing

## 🔄 Continuous Learning
This repository is actively maintained and will be updated as I:
- Learn new machine learning algorithms
- Implement new preprocessing techniques
- Work on additional projects
- Explore advanced ML concepts
- Practice different data science workflows

## 📝 Structure
```
├── README.md
├── requirements.txt
├── data_gathering/           # Data collection notebooks
│   ├── from_api.ipynb       # API data fetching
│   ├── web_scrapping.ipynb  # Web scraping techniques
│   └── with_csv.ipynb       # CSV file handling
├── data_preprocessing/       # Data preprocessing notebooks
│   ├── numerical_ds_preprocessing.ipynb
│   ├── d_standaize_train_test_split.ipynb
│   ├── label_encoding.ipynb
│   ├── handle_missing_values.ipynb
│   ├── handle_imbalanced_dp.ipynb
│   ├── text_ds_preprocessing.ipynb
│   └── text_ds_preprocessing2.ipynb
├── datasets/                 # Data files for projects
│   ├── cardekho.csv
│   ├── diabetes.csv
│   ├── gld_price_data.csv
│   ├── loan_data.csv
│   ├── sonar.all-data.csv
│   ├── Pokemon.csv
│   ├── Sleep_health_and_lifestyle_dataset.csv
│   └── profiles.json
└── projects/                 # Complete ML projects
    ├── diabeties_prediction.ipynb
    ├── Sleep Disorder Prediction.ipynb
    ├── fake_news_prediction.ipynb
    ├── wine_quality_prediction.ipynb
    ├── sonar_rocks_vs_mine_predition.ipynb
    ├── loan_status_prediction.ipynb
    ├── house_price_prediction.ipynb
    ├── house_price_prediction2.ipynb
    ├── price_card_prediction.ipynb
    ├── gold_price_prediction.ipynb
    └── pokemon.ipynb
```

## 🚀 Getting Started
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Open any notebook to explore the learning materials
4. Follow along with the projects to understand different ML concepts

---
*This repository represents my ongoing journey in machine learning. Each notebook contains practical examples and implementations that help solidify theoretical concepts.*