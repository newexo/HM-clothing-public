import numpy as np
import pandas as pd
import os

from hmcollab import directories
from hmcollab import datasets
from hmcollab import splitter
from hmcollab import transactions


def customer_split(dataset, customer_count):
    r = np.random.RandomState(42)

    selected_customers = r.choice(
        dataset.transactions_y.customer_id.unique(), size=customer_count, replace=False
    )

    portion = splitter.CustomerPortion(selected_customers)
    return portion.split(dataset)


def save_main_data(pruned_dataset, base_path):
    customers_fn = directories.qualifyname(base_path, "customers.csv")
    article_fn = directories.qualifyname(base_path, "articles.csv")
    transaction_fn = directories.qualifyname(base_path, "transactions_train.csv")
    pruned_dataset.customers.to_csv(customers_fn, index=False)
    pruned_dataset.articles.to_csv(article_fn, index=False)
    pruned_dataset.transactions.to_csv(transaction_fn, index=False)


def save_relevant_data(relevant_data, base_path, val=False):
        name = "relevant.csv"
        if val:
            name = "relevant_val.csv"
        relevant_fn = directories.qualifyname(base_path, name)
        relevant_data.to_csv(relevant_fn, index=False)


def generate_relevant(transactions_df, days=7, val=False):
    """Relevant transactions are those of interest for prediction (i.e. the most
    recent transactions). It produces a dataframe with all relevant transactions by customer
    separated by a space. The timeframe goes back the number of days specified (from the most 
    recent transaction in the dataset). When val=True, it will use the previous timeframe as 
    the one used when val=False to use as a validation set (instead of the test set).

    Args:
        transactions_df (_type_): transactions dataframe
        days (int, optional): Number of days. Defaults to 7.
        val (bool, optional): True for test set and False for validation set. Defaults to False.

    Returns:
        _type_: dataframe by customer. Columns: "customer_id" and "target"
    """
    transactions_x, transactions_y = splitter.split_by_time(transactions_df, days=days)
    if val:
        _, transactions_y = splitter.split_by_time(transactions_x, days=days)
    
    return datasets.target_to_relevant(transactions_y)
    

def generate_toy(dataset, dir_name="toy", size=10000):
    """
    Creating a toy dataset (also train and test)
    with a specific size (default=10k) of random customer
    The dataset will be save under the data/dir_name directory
    """
    pruned_dataset = customer_split(dataset, size)

    # save it under data/dir_name directory
    print("dir_name:", dir_name)
    print("path:", directories.data(dir_name))
    if not os.path.exists(directories.data(dir_name)):
        os.mkdir(directories.data(dir_name))
    save_main_data(pruned_dataset, directories.data(dir_name))
    return pruned_dataset


def generate_test_data(dataset):
    pruned_dataset = customer_split(dataset, 500)
    save_main_data(pruned_dataset, directories.testdata("fivehundred"))


def main():
    tree = datasets.HMDatasetDirectoryTree()
    dataset = datasets.HMDataset(tree=tree, folds="threesets")
    directory_toy = "toy"

    generate_test_data(dataset)
    toy = generate_toy(dataset, dir_name=directory_toy)
    relevant = generate_relevant(toy.transactions, val=False)
    save_relevant_data(relevant, directories.data(directory_toy), val=False)
    relevant = generate_relevant(toy.transactions, val=True)
    save_relevant_data(relevant, directories.data(directory_toy), val=True)


if __name__ == "__main__":
    main()
