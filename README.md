# Integrated Cardiovascular and Sleep Health Assessment Model (ICSHAM)

## Repository Description
This repository contains 8 files in total:
- heart_data.csv: original patient information related to heart and blood pressure data.
- sleep_data.csv: original patient information related to sleep data.
- DataClean.ipynb: interactive notebook that cleans and preprocesses both datasets above.
- heart_data_clean.csv: cleaned and preprocessed heart dataset.
- sleep_data_clean.csv: cleaned and preprocessed sleep dataset.
- dataLinkage.ipynb: interactive notebook that performs data linkage for both cleaned datasets using Fuzzy matching.
- resultantDataset.csv: linked and merged result dataset that is created through dataLinkage.ipynb.
- FinalProjectModeling.ipynb: interactive notebook that uses ML models to find insights and output results using resultantDataset.csv


## Overview
The Integrated Cardiovascular and Sleep Health Assessment Model (ICSHAM) is a research initiative aimed at developing predictive models to estimate the risk and progression of cardiovascular diseases (CVD) based on sleep quality and patterns. This project leverages machine learning techniques to analyze health data collected from wearable devices, providing insights into how sleep-related metrics can influence cardiovascular health.

## Team Members
- **Daniel Cho** - hjc448@nyu.edu
- **Ali Alshehhi** - aa8148@nyu.edu
- **Kevin Soriano** - ks5885@nyu.edu

## Objectives
1. **Predictive Modeling**: To create a model that uses machine learning to estimate CVD risk from sleep patterns and quality.
2. **Data Integration**: Combine cardiovascular health indicators with detailed sleep metrics to explore correlations.
3. **Health Insights**: Provide actionable insights to improve cardiovascular health through better sleep practices.

## Methodology

### Data Cleaning
To ensure the integrity and uniformity of the data used in our models, we follow these straightforward steps in our data cleaning process:

1. **Standardization**: Convert age measurements from days to years and standardize units across datasets for consistent analysis.

2. **Cleaning**: Identify and remove any outliers or errors in blood pressure readings that fall outside physiological norms (e.g., systolic 50-200 mmHg, diastolic 30-125 mmHg).

3. **Normalization**: Normalize features such as blood pressure and Body Mass Index (BMI) to scale data and improve model performance.

4. **Missing Values**: Fill in or remove missing entries to maintain dataset completeness.

This streamlined approach ensures our datasets are preprocessed effectively, facilitating robust analysis and modeling.

### Data Concantation

To effectively combine our cardiovascular and sleep datasets, we follow these steps to ensure precise and meaningful data integration:

1. **Feature Alignment**: Align common features between datasets, such as age, gender, blood pressure rates, and activity levels.

2. **Fuzzy Matching**: Employ fuzzy matching techniques to identify units with similar, but not necessarily identical, feature values. This method helps in linking related data points across different datasets.

3. **Normalization**: Normalize features to ensure that each has equal weight in the similarity assessment, avoiding bias towards features with larger ranges.

4. **Batch Linkage**: Use batch processing to handle large datasets efficiently, ensuring comprehensive matching without sacrificing computational resources.

5. **Threshold Setting**: Set a similarity threshold to finalize matches, ensuring that only the most relevant data points are merged.

This procedure ensures that our datasets are merged with high accuracy, preserving the integrity of the data while enabling a comprehensive analysis of the interdependencies between sleep patterns and cardiovascular health.


### Data Analysis
Our data analysis process involves several key techniques to develop and evaluate predictive models effectively:

1. **Feature Selection**:
   - **Recursive Feature Elimination (RFE)**: This method helps in reducing the complexity of the model by iteratively removing the least important features.
   - **Principal Component Analysis (PCA)**: Used to transform a large set of variables into a smaller one that still contains most of the information in the large set.

2. **Model Development**:
   - **Random Forest**: A versatile machine learning model that is effective for classification tasks, capable of handling the non-linear relationships between features.
   - **Logistic Regression**: A statistical model that estimates the probability of a binary outcome, such as the presence or absence of cardiovascular disease.
   - **Neural Networks**: Specifically, a fully connected neural network architecture was employed to model complex patterns in the data.

3. **Cross-validation**:
   - To validate the effectiveness and reliability of our models, we employ k-fold cross-validation, which helps in ensuring that our model's performance is not dependent on the way we split up the data.

4. **Performance Metrics**:
   - Models are evaluated using a variety of metrics, including **accuracy**, **sensitivity** (recall), **specificity**, and the **F1-Score**. These metrics provide a comprehensive view of model performance across different aspects.

5. **Iterative Improvement**:
   - Based on the results of our performance metrics, we continuously refine our models. This includes tuning hyperparameters, adding or removing features, and possibly integrating new data sources to enhance predictive capabilities.

Through these methodologies, we aim to develop robust models that can accurately predict cardiovascular disease risk based on sleep patterns and other relevant health indicators.


## Datasets
Our project integrates two key datasets, each containing a range of variables critical for assessing the relationship between sleep patterns and cardiovascular health:

### Cardiovascular Disease (CVD) Dataset
This dataset includes a variety of medical and lifestyle variables necessary for evaluating cardiovascular health:
- **Age and Gender**: Basic demographic information that influences CVD risk.
- **Systolic and Diastolic Blood Pressure**: Important indicators of cardiovascular health.
- **Total Cholesterol and Glucose Level**: Blood chemistry metrics relevant to heart disease.
- **Smoking Status and Alcohol Intake**: Lifestyle factors that significantly impact heart health.
- **Physical Activity**: Reflects general health and is known to affect cardiovascular risk.
- **Body Mass Index (BMI)**: A measure of body fat based on height and weight.
- **Presence of Cardiovascular Diseases**: Indicates whether CVD is already diagnosed.

### Sleep Dataset
This dataset focuses on aspects related to sleep quality and patterns, along with related lifestyle metrics:
- **Sleep Duration and Quality**: Key indicators of sleep health, which have been linked to cardiovascular risk.
- **Stress Levels**: High stress can negatively impact both sleep quality and cardiovascular health.
- **Daily Steps and Physical Activity Level**: Measures of physical activity that relate to both sleep and cardiovascular health.
- **Heart Rate and Blood Pressure**: Recorded during sleep, these variables help correlate sleep quality with cardiovascular conditions.
- **BMI Levels**: Included to observe the impact of obesity on sleep patterns.
- **Occupation and Gender**: Socioeconomic and demographic variables that might influence both sleep and cardiovascular health.

### Integration of Variables
Our analysis combines these datasets by focusing on shared variables to explore the potential correlations between sleep factors and CVD. This integrated approach allows us to employ machine learning techniques to predict CVD risk, examining the role of sleep as a significant health determinant.


## Results
![models](https://github.com/ks5885/Integrated-Cardiovascular-and-Sleep-Health-Assessment-Model-ICSHAM/assets/84752797/8c306e8f-944e-4206-a979-e385719d64e3)
![fcnn](https://github.com/ks5885/Integrated-Cardiovascular-and-Sleep-Health-Assessment-Model-ICSHAM/assets/84752797/cec2b02e-d5c2-4b4b-9f06-6549f2b440ef)

In our project's recent phase, we enhanced the predictive capabilities of our models through meticulous data cleaning, concatenation, and sophisticated model training processes. We employed a range of machine learning models including Random Forest, Decision Trees, Logistic Regression, and K-Nearest Neighbors. After conducting a thorough hyperparameter sweep, the best-performing model, the Random Forest, demonstrated robust predictive accuracy, achieving an impressive 80% accuracy with optimal settings featuring a Gini impurity criterion. Despite achieving moderate success in accurately predicting cardiovascular disease risks, the models exhibited some limitations, particularly reflected in the area under the curve metrics and F1-scores. For example, the area under the curve (AUC) for our primary model was approximately 0.60, indicating a need for further refinement to enhance predictive precision.

To address these limitations, we refined our approach by standardizing data, which aimed to reduce variance and improve model consistency. Additionally, we introduced a Fully Connected Neural Network (FCNN), which marked a significant improvement in our modeling efforts. The FCNN achieved an AUC of around 0.75, demonstrating enhanced capability in distinguishing between positive and negative cases of cardiovascular disease. However, the challenge of low F1-scores persisted, with the neural network model achieving an F1-score around 0.65, suggesting ongoing issues with the balance between precision and recall in our models.

These detailed statistics underline the iterative nature of our work, highlighting both the progress made and the challenges that remain in optimizing the predictive accuracy of our cardiovascular risk models based on sleep patterns.


