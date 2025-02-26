# Financial Loss of Cybersecurity Breaches

## Overview
This project investigates the financial impact of cybersecurity breaches by analyzing historical data and applying advanced machine learning models. It aims to forecast future trends and uncover the key factors driving data compromises.  
Developed as part of the **Modeling the Future Challenge 2025**.



## Data Sources
To ensure robust and comprehensive analysis, we used the following datasets:  
1. **Identity Theft Research Center Data Compromises (2005–2023)**  
   A longitudinal dataset tracking identity theft and data breach incidents.  

2. **U.S. Department of Health and Human Services Office for Civil Rights Data Breach Incidents (2009–2025)**  
   Focuses on data breaches impacting healthcare organizations and protected health information.  

3. **Aggregated Top Data Breaches from Around the World**  
   A global dataset highlighting large-scale breaches across various industries.  



## Models and Methods

### 1. Forecasting Future Data Compromises  
We used time-series modeling to predict future data breach incidents:  
- **ARIMA (Autoregressive Integrated Moving Average)**  
  A statistical model designed to capture historical trends and make accurate forecasts of future compromises.  
  Learn more: [Introduction to ARIMA Models](https://medium.com/analytics-vidhya/a-thorough-introduction-to-arima-models-987a24e9ff71)  



### 2. Analyzing Factors Behind Data Breaches  
We applied machine learning models and interpretability techniques to identify and analyze key contributors to data compromises:  

- **XGBRegressor (Extreme Gradient Boost Regressor)**  
  A powerful gradient-boosting algorithm that identifies factors influencing data breaches.  
  Learn more: [XGBoost Regression In-Depth](https://medium.com/@fraidoonomarzai99/xgboost-regression-in-depth-cb2b3f623281)  

- **Feed Forward Neural Network (FFNN)**  
  A deep learning model combined with feature interpretation techniques to uncover critical insights.
  Learn more about the structure and applications of Feed Forward Neural Networks.  
    [Introduction to Feed Forward Neural Networks](https://towardsdatascience.com/understanding-feedforward-neural-networks-567e8294f0fc)
   
  - **Permutation Analysis**: Used to measure the importance of each feature by observing model performance changes when feature values are shuffled.  
    Learn more: [Understanding Permutation Importance](https://christophm.github.io/interpretable-ml-book/feature-importance.html#permutation)  

  - **SHAP (SHapley Additive exPlanations) Values**: A method for explaining the predictions of machine learning models by assigning each feature a contribution value.  
    Learn more: [SHAP Documentation](https://shap.readthedocs.io/en/latest/)  



## Final Report
All findings and insights are documented in the final report.  
**[Access the Report](https://docs.google.com/document/d/1F9qtb_3Ph9gvQntT3lUEf5PfCRI1ev1EDKWpHZoE7rw/edit?usp=sharing)**  

### Created By:
- **Nandita Raj Kumar**  
- **Shreya Chittipeddi**  

For the **Modeling the Future Challenge 2025**  
