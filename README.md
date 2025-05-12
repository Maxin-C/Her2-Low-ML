# Clinical characteristics, prognosis, and molecular portrait of Her2-low breast cancer patients who received adjuvant chemotherapy: a retrospective, multi-stage study involves two independent cohorts

This repository stores the machine learning code used in the research and the optimal results. Due to privacy concerns regarding the EHR data, we cannot publicly disclose the dataset used in this experiment.  

The meanings of each file are as follows:  

- **ml.ipynb**: The core machine learning code for this project, which includes four methods: Random Forest, Extreme Gradient Boosting (XGBoost), Categorical Boosting (CatBoost), and Multi-Layer Perceptron (MLP). These methods obtain the final model results by searching for the optimal parameter combinations on the dataset.  
- **eval.ipynb**: Used to evaluate the model results on the dataset and generate SHAP plots and AUC-ROC curves.  
- **shapValuePlot.ipynb**: Optimizes the method for plotting SHAP values.  
- **best_param**: Stores the optimal results of each model across different datasets.