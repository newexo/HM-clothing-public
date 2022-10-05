# HM-clothing-project
Python resources for the HM-clothing-project

Data is from [Kaggle contest](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations).

So far we implementend two different recommending models:
+ Popular 
  + top k articles in the dataset in term of transactions
  + Same recommendation to all users
+ Kmeans with KNN
  + kmeans from the transactions of the user using the centroid to find the nearest neighbors (KNN). KNN uses the full articles dataset
  + Recommendations customized by user
