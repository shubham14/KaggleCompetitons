# Fraud_Detection

Research on various deep learning models for Fraud Detection and Anomaly detection. It is also used to submit the IEEE conducted Kaggle challenge on Fraud Detection. https://www.kaggle.com/c/ieee-fraud-detection?utm_medium=email&utm_source=intercom&utm_campaign=ieee-comp-mailer

## Methods Explored:

* XGBoost classifier with SMOTE-ENN for handling label imbalance
* CNN classifier on a FFT response of the feature vector with SMOTE-ENN for handling label imbalance
* VAE to find anomalies by modelling majority class distribution 

## Getting Started

* Clone this repo : git clone https://github.com/shubham14/Fraud_Detection.git
* The different models are present in /models
* Run {model}.py to train and infer

## Prerequisites

* Python 3.6 +
* Pytorch 1.1 + 
* imblanaced-learn (for SMOTEENN)
* matplotlib

## Versioning

I use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Shubham Dash** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Kaggle for providing the data
* Kaggle kernel masters