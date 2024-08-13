<img src="https://github.com/user-attachments/assets/7f252bf0-04f9-4e50-a74e-94b7c36c345e" alt="Header"/> 

# Machine-Learning-with-Python-IBM


## 📄 Summary 
This course provides an overview of the purpose of Machine Learning, and where it applies to the real world. It then covers topics such as supervised vs unsupervised learning, model evaluation, and various useful Machine Learning algorithms. 

To explore the methods of machine learning, and the algorithms involved, many example projects are embarked upon and explored, including health care, banking, telecommunication, and so on.
The [final project](https://github.com/dattesh2507/Machine-Learning-with-Python-IBM/blob/main/Rain%20Prediction%20in%20Australia.ipynb) within this course is the building of a classifier to predict whether there will be rain the following day. It is a classification problem, and KNN, Decision Tree, SVM, and Logistic Regression are all used to determine the best algorithm to use.



## 📑 Main Topics 
- `Introduction to Machine Learning`
  - Examples of machine learning in various industries
  - The steps machine learning uses to solve problems
  - Examples of techniques and Python libraries used 
  - Differences between Supervised and Unsupervised algorithms
  
- `Regression`
  - Simple linear regression
  - Multiple linear regression
  - Non-linear regression
  - Evaluating regression models
  
- `Classification`
  - Comparisons between the different classification methods
  - K Nearest Neighbours (KNN) algorithm
  - Decision Trees
  - Logistic Regression
  - Support Vector Machines
  
- `Clustering`
  - k-Means Clustering
  - Hierarchical Clustering
  - Density Based Clustering
  
- `Linear Classification`
  - Compare and contrast the characteristics of different Classification methods.
Explain the capabilities of logistic regression.
  - Compare and contrast linear regression with logistic regression.
  - Explain how to change the parameters of a logistic regression model.
  - Describe the cost function and gradient descent in logistic regression.
  - Provide an overview of the Support Vector Machine method.
  - Explain how multi-class prediction works.
Apply Classification algorithms on various datasets to solve real world problems.

## 🔑 Key Skills Learned 
- Understanding of various Machine Learning models, such as Regression, Classification, Clustering, and Recommender Systems
- Use of Python for Machine Learning (including Scikit Learn)
- Application of Regression, Classification, Clustering, and Recommender Systems algorithms on various datasets to solve real world problems


# Rain in Australia - Next-Day Prediction Model

## Project Overview
### Data Source
The data used in this project was downloaded from the Kaggle dataset titled [Rain in Australia](https://www.kaggle.com/jsphyg/weather-dataset-rattle-package), which itself was originally sourced from the Australian Bureau of Meteorology's [Daily Weather Observations](http://www.bom.gov.au/climate/dwo/). Additional weather metrics for Australia can be found within the bureau's [Climate Data Online](http://www.bom.gov.au/climate/data/) web app.

### Business Problem
Weather, and humankind's ability to accurately predict it, plays a critical role in many aspects of life. From farmers growing crops to a family planning a weekend vacation to logistical decision making within airlines, rain in particular is highly influential regarding plans. In some instances, the impact of rain can have large financial consequences. As a result, there is a strong interest from a plethora of stakeholders in the ability to accurately forecast rain. The goal of this project is to use the available data to create a next-day prediction model for whether or not it will rain. Such a model could be utilized in a weather app for the benefit of the public at large.

### Repository Structure
```
├── images/          # Exported images of plots
├── saved_models/    # Saved hyperparameter-tuned models for quick access
├── submissions/     # Files used for the project submissions
├── .gitignore
├── LICENSE
├── README.md
├── notebook.ipynb   # Jupyter notebook containing the analysis and models 
└── weatherAUS.csv   # Data on weather conditions in Australia
```

## Exploratory Data Analysis
### Column Definitions
According to the author of the Kaggle dataset and the ["Notes to accompany Daily Weather Observations"](http://www.bom.gov.au/climate/dwo/IDCJDW0000.shtml) published by the Australian Bureau of Meteorology, the meanings and units for each of the columns in the dataset are as follows:

| **Column Name** | **Definition** | **Units** |
| --------------- | -------------- | --------- |
| `Date` | Date of the observation | N/A |
| `Location` | Location of the weather station | N/A |
| `MinTemp` | Minimum temperature in the 24 hours to 9am. Sometimes only known to the nearest whole degree | Degrees Celsius |
| `MaxTemp` | Maximum temperature in the 24 hours to 9am. Sometimes only known to the nearest whole degree | Degrees Celsius |
| `Rainfall` | Precipitation (rainfall) in the 24 hours to 9am. Sometimes only known to the nearest whole millimeter | Millimeters |
| `Evaporation` | "Class A" pan evaporation in the 24 hours to 9am | Millimeters |
| `Sunshine` | Bright sunshine in the 24 hours to midnight | Hours |
| `WindGustDir` | Direction of the strongest wind gust in the 24 hours to midnight | 16 compass points |
| `WindGustSpeed` | Speed of the strongest wind gust in the 24 hours to midnight | Kilometers per hour |
| `WindDir9am` | Direction of the wind at 9am | 16 compass points |
| `WindDir3pm` | Direction of the wind at 3pm | 16 compass points |
| `WindSpeed9am` | Speed of the wind at 9am | Kilometers per hour |
| `WindSpeed3pm` | Speed of the wind at 3pm | Kilometers per hour |
| `Humidity9am` | Relative humidity at 9am | Percent |
| `Humidity3pm` | Relative humidity at 3pm | Percent |
| `Pressure9am` | Atmospheric pressure reduced to mean sea level at 9am | Hectopascals |
| `Pressure3pm` | Atmospheric pressure reduced to mean sea level at 3pm | Hectopascals |
| `Cloud9am` | Fraction of sky obscured by cloud at 9am | Eighths |
| `Cloud3pm` | Fraction of sky obscured by cloud at 3pm | Eighths |
| `Temp9am` | Temparature at 9am | Degrees Celsius |
| `Temp3pm` | Temparature at 3am | Degrees Celsius |
| `RainToday` | Did the current day receive precipitation exceeding 1mm in the 24 hours to 9am | Binary (0 = No, 1 = Yes) |
| `RainTomorrow` | Did the next day receive precipitation exceeding 1mm in the 24 hours to 9am | Binary (0 = No, 1 = Yes) |

### Observations
#### Histograms
![Histograms of data columns](https://github.com/user-attachments/assets/0c549a17-c6a0-4119-84f8-053e0c995081)

## Data Preprocessing
### Missing Values
The primary preprocessing need for this dataset is handling the missing values. Given the strong correlations between certain features, using a multivariate feature imputation method makes sense. While still experimental, the `IterativeImputer` module from `sklearn` is perfect for this use case and appears stable enough. This module...
> "...models each feature with missing values as a function of other features, and uses that estimate for imputation. It does so in an iterated round-robin fashion: at each step, a feature column is designated as output y and the other feature columns are treated as inputs X. A regressor is fit on (X, y) for known y. Then, the regressor is used to predict the missing values of y. This is done for each feature in an iterative fashion, and then is repeated for max_iter imputation rounds. The results of the final imputation round are returned."

Source: [6.4.3. Multivariate feature imputation](https://scikit-learn.org/stable/modules/impute.html#iterative-imputer)

The `IterativeImputer` was applied to all continuous features while categorical features were imputed via `np.random.choice()` with the unique values weighted by their respective probability distributions. 


### Extracting the Month
Rainfall in Australia exhibits seasonality, as shown in the EDA section. Extracting the month value from the `Date` column is a much more useful feature than the full date itself.

## Conclusion
## Results
The best performing model is the  KNN  with an accuracy of approximately 80%. The scores for both the training and testing data were similar, reducing concerns of the model being overfit.

## Next Steps
While this model is a good starting point for rain prediction in Australia, there are several ways in which the model could be improved upon:
- Further hyperparameter tuning
- Engineering new features such as trailing amounts of rain or sunshine
- Collecting additional data from nearby countries (for example, does rain originating in Indonesia or New Zealand have predictive power?)
- Attempting to predict the *amount* of rainfall



