# Project Name
Lead conversion prediction in a mobile app company

## Project Intro/Objective
Lead conversion is a crucial aspect of business, especially for app and website companies. Identifying and predicting which leads are likely to become paying customers is important since those individuals contribute a lot to the company's profit margin. A lead conversion rate of around 11% is deemed to be a good percentage, so every company wants to achieve that and even more, and the best way to do that is to predict these so that the required and appropriate arrangements are made to enable them to complete the process of converting, to keep making profits for the company and business 

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
#### The ability to predict leads that are likely to convert is crucial to this company, so they know which people to target when they first hear about their app. For a lead to convert, there are various processes, the ideal process is, that the lead hears about the app, leads checks out the app, leads reads more about the app elsewhere, leads registers with the app, hence lead converts. During this process, the lead can stop at any time, so it's important to be able to predict the ones that are likely to convert to enable them to complete the converting process successfully. I obtained the dataset from a practice test I was given, and I synthesized the dataset so it doesn't look too much like the original.
### The questions I deemed to explore were:
#### What are the characteristics of leads that converted, is there a trend in their behavior online?
#### Can I predict with a reasonable amount of certainty the leads that are likely to convert, since 11% is the best percentage of leads that actually convert?
#### What's the best way to get this app to the company so they use it to improve their business?

#### Some of the challenges faced were:
#### Only a little under 5% of leads converted with this app, so the best way to balance the data appropriately so the ML algorithm trains on it without bias
#### Which model to use that gives high performance, and also explainability

## Getting Started
1. Clone this repo (for help see this [tutorial](https://help.github.com/articles/cloning-a-repository/)).
2. For users, go to https://share.streamlit.io/joamps/lead_conversion_prediction_for_a_mobile_app_company/main/frontend-streamlit/streamlit_app.py to test the app out

## Results
### Analytics
The first and important thing was to look at the characteristics of leads that converted, 
![Lead country of origin](https://github.com/JoAmps/Lead_conversion_prediction_for_a_mobile_app_company/blob/main/leads_per_country.png)
Most leads reside in the United States, with over 98% of leads, but only less than 5% of them converted, compared to the Uk which accounted for less than 1% of all leads, had the most leads converting, with around 5.27% of leads converting, which suggests leads registering in the UK are to be targeted more since they tend to convert the most
![Lead source ](https://github.com/JoAmps/Lead_conversion_prediction_for_a_mobile_app_company/blob/main/leads_per_sources.png)
Facebook is the source that most leads redirect from, with almost half of all leads redirecting from there, but only 3.73% of leads convert from there. Medium redirection accounts for the most lead conversion with 6.76%, even though less than 1% of all leads redirect from there, so leads that redirect from medium should be given the most attention
![Lead devices](https://github.com/JoAmps/Lead_conversion_prediction_for_a_mobile_app_company/blob/main/lead_devices.png)
The phone is the device most leads use to register for the app, with almost 94% of all leads registering with it, but only 4.43% of such leads converted. The desktop has the highest conversion rate, with around 6.69% of all leads converting when they used the desktop to sign up, but only 5% of all leads signed up using the desktop, this could be due to the fact that leads have a bigger screen to look at the app and see all its features, and its easier to navigate using the desktop, example clicking a link on the app could just redirect in a new tab, keeping the other tab open, whereas, in the phone, the old tab probably would be closed. So the best option here is to target leads using desktops as they are more likely to convert.

### Model building and Evaluation
In building the models, 4 algorithms were used, in which weights and biases were used to track their experiments, and the best performing model was the random forest algorithm, so that's the model was chosen for further development.
![Experiment tracking](https://github.com/JoAmps/Lead_conversion_prediction_for_a_mobile_app_company/blob/main/plots/wandb_experiment_trackings.png)
The performance of the model was evaluated on 5 metrics, accuracy(which isn't a good metric for unbalanced datasets), recall, precision, f1score, and roc_auc. The model does not do quite that well in general, achieving scores of around 60% for the metrics, as it's hard to predict leads that would convert as leads have very unique characteristics, but it makes an effort to predict them. The recall is the most important metric here, as we are interested in predicting all leads that converted, even if it means having a few leads that did not convert predicted as converted, so here having a few more false positives is acceptable. The main thing here is to minimize false negatives, leads that converted but predicted to not convert, as having more false negatives would mean missing out on key leads that can potentially make the company profit. 
The Roc curve with the AUC score can be seen below,
![roc_curve](https://github.com/JoAmps/Lead_conversion_prediction_for_a_mobile_app_company/blob/main/plots/roc_curve.png)
Every lead has different characteristics, where they are based, the device they use, the time they registered, etc. Still, there's a pattern in which leads that converted or leads that did not take, so in coming up with the predictions, the model identified that some of the features were more important than others in predicting if leads would convert or not, below we can see the feature importance. The most important feature in predicting lead conversion is if the lead registered and signed up with the same on the same day, which makes sense as those leads have the highest interest in the app. Generally, the length of time between registering and subscribing is the most important in predicting leads that would convert. The country where the lead is based does not seem to be a very distinguishing feature, as no country makes the top 15 important features, but it still is important to predict lead conversion. The feature that was the least important is if the lead uses a set-top box to access the app, as it is an outmoded device
![Feature importance](https://github.com/JoAmps/Lead_conversion_prediction_for_a_mobile_app_company/blob/main/plots/feature_importance.png)


### WebApp
Fast API was used to build the backend, which takes the trained model and outputs a prediction, and streamlit was used as the frontend, to display the results in a nice UI that company leaders can use to access their leads. The
The web app can be accessed here --> https://share.streamlit.io/joamps/lead_conversion_prediction_for_a_mobile_app_company/main/frontend-streamlit/streamlit_app.py. Ideally, it's supposed to be for only the company management to access and use it to improve their business, but since this is a personal project, any user can get access and view it.
For developers to use and run this application at their end, the whole app is containerized using docker
