# Infrdd_Assignment
# INFRDD Assignment

## **Introduction**

The dataset comprises multiple dataframes, each corresponding to an image of a form. Each piece of information extracted from the form is represented as a token and occupies a row in the dataframe. The available features include:

+ **start_index** and **end_index**: These columns indicate the position of the token as if the entire image were flattened into a single line.
+ Four columns represent the **x** and **y** coordinates of the top-left and bottom-right corners of the token.
+ **transcript**: This column contains the information captured within the token.
+ **field**: This column serves as the label, identifying the type of information contained in the token.

## **Exploratory Data Analysis and Feature Engineering**

This section draws significant inspiration from the Exploratory Data Analysis conducted in Luis Fernando Torres's project on **Wine Quality: EDA, Prediction, and Deployment**.

### **Key Observations:**

+ The data is highly skewed, with the **“OTHER”** label significantly outnumbering all other labels combined.
+ Introduced additional features: **File No.**, **index_len**, **x_center**, and **y_center**.
+ Removed the **transcript** feature due to its diverse content and lack of time for NLP techniques.
+ Analyzed correlation heatmaps to identify significant features.

## **Model Pipeline**

Based on the preceding analysis, I have implemented a two-tier classification system:

+ **First Model**: Processes the entire dataset to classify entries as either **“OTHER”** or **“not other.”**
+ **Second Model**: Trained on the filtered dataset to identify the actual label of entries deemed relevant.

### **Workflow:**

1. Merge the entire dataset into a single dataframe.
2. Perform feature engineering on the columns.
3. Create a copy of the dataframe for the distinct requirements of both models.
4. Convert all relevant labels to **“not other”** for the first model and undersample the data to address skewed class distribution.
5. Drop all entries labeled as **“OTHER”** for the second model and train both models on their respective datasets.

## **Hyperparameter Tuning**

+ Employed **Stratified K-Fold Cross Validation** to determine the optimal **n_estimators** parameter, resulting in:
  + **250** estimators for both models.
  + Average accuracy of approximately **96.57%** for the first model and around **94.96%** for the second model.

## **Results**

+ Selected **n_estimators** = **250** and trained the entire dataset, achieving **100% accuracy** on a sample from the validation dataset.
+ The entire code for this section is available in **main.ipynb**.

## **Dependencies**

The project relies on the following Python libraries:

+ **pandas**
+ **numpy**
+ **scikit-learn**
+ **matplotlib**
+ **seaborn**

## **How to Run the Code**

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Infrdd_Assignment.git
   

**Conclusion**
This project showcases a comprehensive approach to exploring and modeling a dataset consisting of form data. Through exploratory data analysis, feature engineering, and a two-tier classification model, I aimed to address the challenges posed by skewed data and optimize model performance.
 
