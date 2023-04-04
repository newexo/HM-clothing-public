# HM-clothing-project
Python resources for the HM-clothing-project

### Data
Data is from [Kaggle contest](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations).
From 2018-09-20 to 2020-09-22 (two years).

**transactions full**
+ Transactions shape:  (31788324, 5)
+ Articles with transactions:  104547
+ Unique customers:  (1362281,)
+ articles with at least 300 transactions: 25,312

**transactions toy**
+ Transactions shape:  (252406, 5)
+ Articles with transactions:  47908
+ Unique customers:  (10899,)
+ articles with at least 50 transactions: 252

### Models
So far we implementend two different recommending models:
+ **Popular** 
  + top k articles in the dataset in term of transactions
  + Same recommendation to all users
+ **Kmeans with KNN**
  + kmeans from the transactions of the user using the centroid to find the nearest neighbors (KNN). KNN uses the full articles dataset
  + Recommendations customized by user

### Split strategies
+ **Standard (leave last week)**: Split transactions in three using a cutoff date (default days=7)
+ **Folds** : Split customers in two or three subsets and each subset is also split in the standard fashion