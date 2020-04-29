# United States Forest Fires
It seems like in the past few years it seems like there is always news of a wild fire. One question is what causes these fires and if machine learning can predict the cause of the fires. It would be usefull to know the cause of fires from the fact that usually wild fires have a cost to them even in the form of property damage. Insurance companies will benefit the most from knowing the cause of a fire right away since they can either get money from whoever caused the fire or know if they cant get any money if it was natural.

# Data Set
The Data set came from <a href='https://www.kaggle.com/rtatman/188-million-us-wildfires'>
    Kaggle
</a>. The data set is a publication from national Fire Program Analysis (FPA) system. It contains 1.88 million observations from 1992-2015 and stored in a sql database. I sampled 100k from the database. 

# Work Flow
## EDA and Data Cleaning
 
 First I had to access my data from an sqlite data file. I then looked into the columns of the data and explored what the meaning of certain columns are. I explored which states had the most wild fires which the top three were California, Arizona, and Oregon. The most interesting I thought was Idaho because I dont hear much about wild fires in that state. I then explored into who owns the properties that the fire was at and it came out to majority government owned land. I then looked into the cause of fires and total observations. The majority of the causes were Lightining with the next greatest cause are campfires though the difference between lighting and campfire is high. 
 
## Modeling
The first model I ran was a random forest with no parameter tuning. 
I then ran a random forest with a grid search to look for which parameters will give me the best scores. The Random forest with no hyper parameter tuning had better recall, precision, and f1 score that is better than all other non hyper parameter tuned models.
I then ran an adaboost model with no hyper parameter tuning and a second AdaBoost with a grid search as well.
I then ran an XGBoost model and followed the same steps as my previous models.
 
# Conclusion

# Next Steps
Continue to hyper parameter already made models to get better scores. I want to run a nueral network model to see if it can do a better job classifying my target variable (which the causes of fires). I want to see if can sample more data for other causes of fires that is less than lightining to see if having more observations for models can help improve my scores.