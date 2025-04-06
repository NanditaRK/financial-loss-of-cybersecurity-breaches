# Financial Loss of Cybersecurity Breaches

## Overview
This project investigates the financial impact of cybersecurity breaches by analyzing historical data and applying advanced machine learning models. It aims to forecast future trends and uncover the key factors driving data compromises.  
Developed as part of the **Modeling the Future Challenge 2025**.



## Data Sources
To ensure robust and comprehensive analysis, we used the following datasets:  

### 1. The Identity Theft Research Center Data Compromises 2005 - 2023 [1]

- **Data Categories:** define historical trends, projecting future trends, defining the frequency of potential outcomes
- **Variables:** year, compromises, and individuals impacted in millions

### 2. Ponemon Institute and IBM Cost of a Data Breach Reports 2008-2024 [2]

- **Data Category:** define historical trends, projecting future trends, defining the frequency of potential outcomes, separating potential outcomes
- **Variables:** year, average cost of data breach (in millions), average per record cost of a data breach (in USD)

### 3. The Identity Theft Research Center Data Breaches and Incident Reports 2016 - 2023[3]

- **Data Category:** projecting future trends, separating potential outcomes
- **Variables:** sector, victims impacted, state, breached reported date, types of record exposed, date of breach

### 4. U.S. Department of Health and Human Services Office for Civil Rights Data Breach Incidents 2009 - 2025 [4]
   
- **Data Category:** defining the frequency of potential outcomes, separating potential outcomes
- **Variables:** name of covered entity, state, covered entity type, individuals affected, breach submission date, type of breach location of breached information, business associate present, web description of breach

### 5. The VERIS Risk Database 2006 - 2025[5]
   
- **Data Category:** define historical trends, projecting future trends, defining the frequency of potential outcomes, separating potential outcomes, defining severity
- **Variables:** over 2000 different variables because their columns were represented in one hot encoded style (all the variables and info about this dataset can be found at verisframework.org

### 6. Aggregated Top Data Breaches from Around the World [6]
   
- **Data Category:** defining the frequency of potential outcomes, separating potential outcomes, define severity
- **Variables:** organization, records lost, year, date, sector, method, data sensitivity




## Models and Methods

### 1. Forecasting Future Data Compromises  
We used time-series modeling to predict future data breach incidents:  
- **ARIMA (Autoregressive Integrated Moving Average)**  
  A statistical model designed to capture historical trends and make accurate forecasts of future compromises.  
  Learn more: [Introduction to ARIMA Models](https://medium.com/analytics-vidhya/a-thorough-introduction-to-arima-models-987a24e9ff71)  

### 2. Predicting the Loss of an Individual Data Breach  
We created our our feed forward neural network architecture to predict the individual financial loss of an incident:
- **Feed Forward Neural Network (FFNN)**  
  A deep learning model combined with feature interpretation techniques to uncover critical insights.
  Learn more about the structure and applications of Feed Forward Neural Networks.  
    [Introduction to Feed Forward Neural Networks](https://towardsdatascience.com/understanding-feedforward-neural-networks-567e8294f0fc)

### 3. Analyzing Factors Behind Data Breaches  
We applied machine learning models and interpretability techniques to identify and analyze key contributors to data compromises:  

- **XGBRegressor (Extreme Gradient Boost Regressor)**  
  A powerful gradient-boosting algorithm that identifies factors influencing data breaches.  
  Learn more: [XGBoost Regression In-Depth](https://medium.com/@fraidoonomarzai99/xgboost-regression-in-depth-cb2b3f623281)  


 ### Techniques Used Across Multiple Models    
  - **Permutation Analysis**: Used to measure the importance of each feature by observing model performance changes when feature values are shuffled.  
    Learn more: [Understanding Permutation Importance](https://christophm.github.io/interpretable-ml-book/feature-importance.html#permutation)  

  - **SHAP (SHapley Additive exPlanations) Values**: A method for explaining the predictions of machine learning models by assigning each feature a contribution value.  
    Learn more: [SHAP Documentation](https://shap.readthedocs.io/en/latest/)
 
  - **Grid Search**: A method to find the best hyperparameters for your model by traversing through all combination of hyperparameters and comparing model evaluations. 
    Learn more: [Grid Search Deep Dive](https://medium.com/@data-overload/unlocking-model-optimization-a-deep-dive-into-grid-search-7805b0a33133)

## Final Report
All findings and insights are documented in the final report.  
**[Access the Report](https://docs.google.com/document/d/1F9qtb_3Ph9gvQntT3lUEf5PfCRI1ev1EDKWpHZoE7rw/edit?usp=sharing)**  

### Created By:
- **Nandita Raj Kumar**  
- **Shreya Chittipeddi**  

For the **Modeling the Future Challenge 2025**  
