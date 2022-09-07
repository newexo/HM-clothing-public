# Data directory
This directory will contain training and test files. Most files in it will not be checked in to git.

target_set_7d75481u.csv 
+ Transactions from the last 7 days by customer from the full dataset
+ Columns: customer_id,last_7d
+ 75,481 unique customers

transactions_toy.csv
+ Subset with all transactions from < 1% of customers 
+ Transactions shape:  (252406, 5)
+ Unique customers:  (10899,)

toy_relevant_set.csv
+ Transactions from the last 7 days by customer from the toy dataset (transactions_toy.csv)
+ Columns: customer_id,target
+ Transactions: 2035
+ Unique customers with transactions in the last 7 days: 578