# United States Forest Fires
It seems like in the past few years it seems like there is always news of a wild fire. One question is what causes these fires and if machine learning can predict the cause of the fires. It would be usefull to know the cause of fires from the fact that usually wild fires have a cost to them even in the form of property damage. Insurance companies will benefit the most from knowing the cause of a fire right away since they can either get money from whoever caused the fire or know if they cant get any money if it was natural.

# Repo Navigation

- <a href='https://github.com/MixMaster1/U.S.-WildFires/blob/master/Forest%20Fires%20in%20the%20U.S.pdf'>
    Forest Fires in the U.S.pdf
</a>: PDF file of google slide presentation

- <a href='https://github.com/MixMaster1/U.S.-WildFires/blob/master/Modeling%20and%20EDA.ipynb'>
    Modeling and EDA.ipynb
</a> Jupyter Notebook of all work 

- <a href='https://github.com/MixMaster1/U.S.-WildFires/blob/master/README.md'>
    README.md
</a>

- <a href='https://github.com/MixMaster1/U.S.-WildFires/blob/master/requirements.txt'>
    requirements.txt
</a>: These are the packages that were in my enviroment when was creating this analysis. Use this file to create a virtual enviroment in order to run this notebook.

# Data Set
The Data set came from <a href='https://www.kaggle.com/rtatman/188-million-us-wildfires'>
    Kaggle
</a>. The data set is a publication from national Fire Program Analysis (FPA) system. It contains 1.88 million observations from 1992-2015 and stored in a sql database. I sampled 100k from the database. 

# Work Flow
## EDA and Data Cleaning
 
 First I had to access my data from an sqlite data file. I then looked into the columns of the data and explored what the meaning of certain columns are. I explored which states had the most wild fires which the top three were California, Arizona, and Oregon. The most interesting I thought was Idaho because I dont hear much about wild fires in that state. I then explored into who owns the properties that the fire was at and it came out to majority government owned land. I then looked into the cause of fires and total observations. The majority of the causes were Lightining with the next greatest cause are campfires though the difference between lighting and campfire is high. I dropped multiple columns that I felt like had no importances for what I was attempting to solve or was just basically column that was a duplicate of another. For example OWNER_CODE and OWNER_DESCR are basically the same columns of information but one is in float data and another in strings. I kept the float since the strings cannot be put into models. There were other columns as well that just referenced what agency is preparing the fire report or another columns which just gave information of what they named the fire. These would not be important to me while modeling since I do not want my model to associate names of fires or agencys to a cause of a fire. Once I dropped my columns I then got rid of any NA values that were present in any rows I may have and which there were 171 of them.
 
## Modeling tuned to accuracy
The first model I ran was a decision tree to look into what features it finds important and having a baseline to tell how well my future model are performing.
I ran a random forest with a grid search and the top three feature importance for the best estimator were Discovery_DOY which is when a fire is discovered on the specific day of the year, Discovery_Time which is when the fire discovered in military time, and Discovery_Date which is a specific day since julian date (January 1, 4713). For the best estimator its accuracy was a .684 which compared to the decision tree it was 
I then ran an adaboost model with no hyper parameter tuning and a second AdaBoost with a grid search as well.
I then ran an XGBoost model and followed the same steps as my previous models. XGBoost was ran before the precision models but was still valueable to look at how it was doing in accuracy which it was doing better than all other models tuned to accuracy. Its precision was worse than the hyper parameter random forest tuned to precision but did better than every other models' precision score. The precision score for the XGBoostClassifier model is a .446. and the accuracy is a .685.


## Models tuned to precision
In my modeling I want to focus on Precision metric since in this bussiness case it would be more important to have my models correctly predict a cause of a fire for an insurance company to minimize the cost of investigating a cause of a fire. I changed the scoring parameter in the grid searches. 
```
grid_rfc_prec = GridSearchCV(rfc_pmtuned_prec, rfc_param_prec, cv=3,
                        scoring= "precision_macro")
```
I followed the same steps as all my previous models but then evaluated how well the RandomForest model did compared to the DecisionTree and also the last hyper parameter RandomForest model. The precision score for my RandomForest tuned to precision is a .57 compared to the decision tree .29 and the RandomForest tuned to accuracy is a .45. The AdaBoost model tuned to precision did worse though than the RandomForest Tuned to precision with a .24 decrease. This models precision score is a .33 while the RandomForest is a .57.

# Conclusion
My RandomForest was my best performing model with a precision score of a .57 and is a 28 percent increase in precision compared to the baseline. Models can definitely help the case of figuring out what may have caused a fire providing location and time the fires started and stopped. There is more work to get a better scoring with my models. The use of machine learning can help minize the cost a company can waste on highering investigators to find out what caused a fire and narrow where to focus resources. 
# Next Steps
 I want to run a nueral network model to see if it can do a better job classifying my target variable. I want to see if can sample more data for other causes of fires that is less than lightining to see if having more observations can help improve my scores. The use of PCA (principal component analysis) maybe useful here to reduce dimesionality. I want to see if using that can help improve my precision scores across my models. I want to see if the use of clustering may also be useful to improve my scores. I will see if I can just change my target varaiable to human causes and nature causes of wildfire and see if that will help improve my scores which would still be useful information for insurers to see if it is worth investigating. Create an ineractive map for users can see previous fires and information about those fires. Additionaly, I need to build my own grid search since the sklearn version of grid search was not getting along well with the XGBoostClassifier.