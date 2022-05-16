# Project Name
Lead conversion prediction in a mobile app company

## Project Intro/Objective
Lead conversion is a crucial aspect of business, especially for app and website companies. Identifying and predicting which leads are likely to become paying customers is important, since those individuals contribute a lot to the company's profit margin. A lead conversion rate of around 11% is deemed to be a good percentage, so every compny wants to achieve that and even more, and the best way to do that is to predict these, so that the required and appropriate arrangements are made to enable them complete the process of converting, to keep making profits for the company and business 

### Methods Used
* Data exploration/descriptive statistics
* Data processing/cleaning
* Inferential Statistics
* Machine Learning
* Data Visualization
* Experiment tracking
* Testing
* Deployment
* Containerization
* Continous integration/Continous deployment(CI/CD)

### Technologies
* Python
* Various python libraries for data science and machine learning
* Streamlit for frontend
* Docker for containerization
* Weights and biases for experiment tracking
* Fast API for backend
* Visual studio code, jupyter
* Git
* Unit Testing(pytest)
* Github actions for CI/CD

## Project Description
#### The ability to predict leads that are likely to convert is crucial to this company, so they know which people to target wehn they first hear about their app. For a lead to convert, there are various processes, the ideal process is,lead hears about app, leads checks out app, leads reads more about the app elsewhere, leads registers with the app, hence lead converts. During this process, the lead can stop at anytime, so its important to be able to predict the ones that are likley to convert to enable them complete the converting process succesfully. I obtained the dataset from a practise test i was given, and i synthesized the dataset so it doesnt look too much like the original.
### The questions i deemed to explore were:
#### What are the characteristics of leads that converted, is there a trend in their behaviour online?
#### Can i predict with a reasonable amount of certainty the leads that are likely to convert, since 11% is the best percentage of leads that actually convert?
#### Whats the best way to get this app to the company so they use to improve their business?

#### Some of the challenges faced were:
#### Only a little under 5% of leads converted with this app, so best way to balance the data appropriately so the ML algorithm trains on it without bias
#### Which model to use that gives high performance, and also explainability

## Getting Started
1. Clone this repo (for help see this [tutorial](https://help.github.com/articles/cloning-a-repository/)).
2. For users, go to https://share.streamlit.io/joamps/lead_conversion_prediction_for_a_mobile_app_company/main/frontend-streamlit/streamlit_app.py to test the app out

## Results
The first and important thing was to look at the characteristics of leads that converted, 
![Lead country of origin](https://github.com/JoAmps/Lead_conversion_prediction_for_a_mobile_app_company/blob/main/leads_per_country.png)
Most leads reside in the United states, with over 98% of leads, but only less than 5% of them converted, compared to the Uk which accounted for less than 1% of all leads, had the most leads converting, with around 5.27% of leads converting, which suggests leads registering in the UK are to be targeted more, since they tend to convert the most
![Lead source ](https://github.com/JoAmps/Lead_conversion_prediction_for_a_mobile_app_company/blob/main/leads_per_sources.png)
Facebook is the source that most leads redirect from, with almost half of all leads redirecting from there, but only 3.73% of leads convert from there. Medium redirection acounts for the most lead conversion with 6.76%, even though less than 1% of all leads redirect from there, so leads that redirect from medium should be given the most attention
![Lead devices](https://github.com/JoAmps/Lead_conversion_prediction_for_a_mobile_app_company/blob/main/lead_devices.png)
Phone is the device most leads use to register for the app, with almost 94% of all leads registering with it, but only 4.43% of such leads converted. Desktop has the highest conversion rate, with around 6.69% of all leads converting when they used desktop to sign up, but only 5% of all leads signed up using the desktop, this could be due to the fact that leads have a bigger screen to look at the app and see all its features, amd its easier to navigate using the desktop, example clicking a link on the app could just redirect in a new tab, keeping the other tab open, where as in phone,the old tab probably would be closed. So best option here is to target leads using desktops as they are more likely to convert

In building the models, 4 algorithms were used, in which weights and biases was used to track their experiments, and the best performing model was the random forest algorithm, so thats the model chosen for further development.
![Experiment tracking](https://github.com/JoAmps/Lead_conversion_prediction_for_a_mobile_app_company/blob/main/plots/wandb_experiment_trackings.png)
The performance of the model was evaluated on 5 metrics, accuracy(which isnt a good metric for unbalanced datasets), recall, precision, f1score and roc_auc. The model does not do quite that well in general, achieving scores of around 60% for the metrics, as its hard to predict leads that would convert as leads have very unique characteristics, but it makes an effort to predict it. The Roc curve with the AUC score cna be seen below,
![Experiment tracking](https://github.com/JoAmps/Lead_conversion_prediction_for_a_mobile_app_company/blob/main/plots/roc_curve.png)
Fast api was used to build the backend, which tales the trained model and outputs a prediction, and stramlit was used as the frontend, to display the results in a nice UI that company leaders can use to access their leads.
