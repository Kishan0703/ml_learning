# ML Learning Journey 🚀

This repository contains my machine learning learning journey and will be continuously updated as I learn and implement new concepts.

## 📚 Learning Topics Covered

### 📥 Data Gathering

- [**From API**](data_gathering/from_api.ipynb) - Fetching data from REST APIs
- [**Web Scraping**](data_gathering/web_scrapping.ipynb) - Extracting data from websites
- [**From CSV**](data_gathering/with_csv.ipynb) - Working with CSV files

### 🔧 Data Preprocessing

- [**Numerical Data Preprocessing**](data_preprocessing/numerical_ds_preprocessing.ipynb) - Preprocessing numerical datasets
- [**Handle Missing Values**](data_preprocessing/handle_missing_values.ipynb) - Techniques for dealing with missing data
- [**Handle Imbalanced Data**](data_preprocessing/handle_imbalanced_dp.ipynb) - Methods for handling imbalanced datasets
- [**Text Data Preprocessing**](data_preprocessing/text_ds_preprocessing.ipynb) - Text cleaning and feature extraction
- [**Text Data Preprocessing 2**](data_preprocessing/text_ds_preprocessing2.ipynb) - Advanced text processing techniques

### ⚙️ Feature Engineering

#### Feature Scaling
- [**Standardization**](feature%20engineering/feature%20scaling/standaization.ipynb) - Z-score normalization (mean=0, std=1)
- [**Normalization**](feature%20engineering/feature%20scaling/normalization.ipynb) - Min-Max scaling

#### Encode Categorical Data
- [**One Hot Encoding**](feature%20engineering/encode%20categorical%20data/one_hot_encoding.ipynb) - Converting categorical to binary columns
- [**Ordinal & Label Encoding**](feature%20engineering/encode%20categorical%20data/ordinal_and_label_encoding.ipynb) - Converting categories to integers

#### Encoding Numerical Data
- [**Binarization**](feature%20engineering/encoding%20numerical%20data/binarization.ipynb) - Converting numerical data to binary
- [**Discretization**](feature%20engineering/encoding%20numerical%20data/discritization.ipynb) - Binning continuous variables

#### Transformers
- [**Column Transformer**](feature%20engineering/transformer/column_transformer.ipynb) - Apply different transformations to different columns
- [**Function Transformer**](feature%20engineering/transformer/function_transformer.ipynb) - Custom transformations using functions
- [**Power Transformer**](feature%20engineering/transformer/power_transformer.ipynb) - Box-Cox and Yeo-Johnson transformations

#### Pipelines
- [**Titanic Without Pipeline**](feature%20engineering/pipelines/titanic_without_pipeline.ipynb) - Manual preprocessing steps
- [**Titanic With Pipeline**](feature%20engineering/pipelines/titanic_with_pipeline.ipynb) - Using sklearn Pipeline
- [**Predict Without Pipeline**](feature%20engineering/pipelines/predict_without_pipeline.ipynb) - Manual prediction workflow
- [**Predict With Pipeline**](feature%20engineering/pipelines/predict_with_pipeline.ipynb) - Prediction using Pipeline

### 🤖 Machine Learning Projects

#### Classification
- [**Diabetes Prediction**](projects/diabeties_prediction.ipynb) - Binary classification for diabetes prediction
- [**Sleep Disorder Prediction**](projects/Sleep%20Disorder%20Prediction.ipynb) - Multi-class classification for sleep disorders
- [**Fake News Detection**](projects/fake_news_prediction.ipynb) - NLP classification for fake news
- [**Wine Quality Prediction**](projects/wine_quality_prediction.ipynb) - Multi-class classification for wine quality
- [**Sonar Rocks vs Mine**](projects/sonar_rocks_vs_mine_predition.ipynb) - Binary classification using sonar data
- [**Loan Status Prediction**](projects/loan_status_prediction.ipynb) - Predicting loan approval status

#### Regression
- [**House Price Prediction**](projects/house_price_prediction.ipynb) - Regression model for house prices
- [**House Price Prediction 2**](projects/house_price_prediction2.ipynb) - Advanced regression techniques
- [**Car Price Prediction**](projects/price_card_prediction.ipynb) - Regression model for car prices
- [**Gold Price Prediction**](projects/gold_price_prediction.ipynb) - Time series prediction for gold prices

#### Exploratory Data Analysis
- [**Pokemon Data Analysis**](projects/pokemon.ipynb) - Comprehensive EDA on Pokemon dataset

## ️ Technologies Used
- **Python** - Primary programming language
- **pandas** - Data manipulation and analysis
- **scikit-learn** - Machine learning algorithms and preprocessing
- **numpy** - Numerical computations
- **matplotlib/seaborn** - Data visualization
- **requests/BeautifulSoup** - Web scraping and API calls

## 📁 Structure
```
├── README.md
├── requirements.txt
├── data_gathering/
│   ├── from_api.ipynb
│   ├── web_scrapping.ipynb
│   └── with_csv.ipynb
├── data_preprocessing/
│   ├── numerical_ds_preprocessing.ipynb
│   ├── handle_missing_values.ipynb
│   ├── handle_imbalanced_dp.ipynb
│   ├── text_ds_preprocessing.ipynb
│   └── text_ds_preprocessing2.ipynb
├── feature engineering/
│   ├── feature scaling/
│   │   ├── standaization.ipynb
│   │   └── normalization.ipynb
│   ├── encode categorical data/
│   │   ├── one_hot_encoding.ipynb
│   │   └── ordinal_and_label_encoding.ipynb
│   ├── encoding numerical data/
│   │   ├── binarization.ipynb
│   │   └── discritization.ipynb
│   ├── transformer/
│   │   ├── column_transformer.ipynb
│   │   ├── function_transformer.ipynb
│   │   └── power_transformer.ipynb
│   └── pipelines/
│       ├── titanic_without_pipeline.ipynb
│       ├── titanic_with_pipeline.ipynb
│       ├── predict_without_pipeline.ipynb
│       └── predict_with_pipeline.ipynb
└── projects/
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

---
*This repository represents my ongoing journey in machine learning.*