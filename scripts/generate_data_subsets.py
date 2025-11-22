import numpy as np
import os

from hmcollab import directories
from hmcollab import datasets
from hmcollab import splitter


def customer_split(dataset, customer_count):
    r = np.random.RandomState(42)

    selected_customers = r.choice(
        dataset.transactions_y.customer_id.unique(),
        size=customer_count,
        replace=False
    )

    portion = splitter.CustomerPortion(selected_customers)
    return portion.split(dataset)


# ----------------------------------------------------------------------
# Save main dataframes as PARQUET instead of CSV
# ----------------------------------------------------------------------
def save_main_data(pruned_dataset, base_path):
    # original filenames (CSV), we will convert them to .parquet
    customers_fn = directories.qualifyname(base_path, "customers.parquet")       ### CHANGED
    article_fn   = directories.qualifyname(base_path, "articles.parquet")        ### CHANGED
    transaction_fn = directories.qualifyname(base_path, "transactions_train.parquet")  ### CHANGED

    pruned_dataset.customers.to_parquet(customers_fn, index=False)               ### CHANGED
    pruned_dataset.articles.to_parquet(article_fn, index=False)                  ### CHANGED
    pruned_dataset.transactions.to_parquet(transaction_fn, index=False)          ### CHANGED


def save_full_dataset_as_parquet(dataset, dir_name="full"):
    """
    Save the full (unpruned) dataset as parquet files.
    This creates a 'full' directory parallel to 'toy', 'toy_1k', etc.
    """
    path = directories.data(dir_name)
    if not os.path.exists(path):
        os.mkdir(path)

    # Use the same naming as toy sets
    customers_fn    = directories.qualifyname(path, "customers.parquet")
    articles_fn     = directories.qualifyname(path, "articles.parquet")
    transactions_fn = directories.qualifyname(path, "transactions_train.parquet")

    # Full datasets
    dataset.customers.to_parquet(customers_fn, index=False)
    dataset.articles.to_parquet(articles_fn, index=False)

    # Combine X + Y transactions to get full training transaction set
    full_transactions = (
        dataset.transactions_x
        .append(dataset.transactions_y, ignore_index=True)
    )

    full_transactions.to_parquet(transactions_fn, index=False)

    print(f"Saved full dataset to: {path}")


# ----------------------------------------------------------------------
# Save relevant datasets as PARQUET
# ----------------------------------------------------------------------
def save_relevant_data(relevant_data, base_path, val=False):
    name = "relevant.parquet" if not val else "relevant_val.parquet"            ### CHANGED
    relevant_fn = directories.qualifyname(base_path, name)
    relevant_data.to_parquet(relevant_fn, index=False)                           ### CHANGED


def generate_relevant(transactions_df, directory_toy=None, days=7, val=False):
    transactions_x, transactions_y = splitter.split_by_time(transactions_df, days=days)
    if val:
        _, transactions_y = splitter.split_by_time(transactions_x, days=days)

    relevant = datasets.target_to_relevant(transactions_y)

    if directory_toy is not None:
        save_relevant_data(relevant, directories.data(directory_toy), val=False)

    return relevant


def generate_toy(dataset, dir_name="toy", size=10000):
    pruned_dataset = customer_split(dataset, size)

    # ensure directory exists
    path = directories.data(dir_name)
    print("dir_name:", dir_name)
    print("path:", path)
    if not os.path.exists(path):
        os.mkdir(path)

    save_main_data(pruned_dataset, path)
    return pruned_dataset


def generate_test_data(dataset):
    pruned_dataset = customer_split(dataset, 500)
    save_main_data(pruned_dataset, directories.testdata("fivehundred"))


def generate_toy_and_relevant(dataset, directory_toy, size):
    toy = generate_toy(dataset, dir_name=directory_toy, size=size)
    generate_relevant(toy.transactions, directory_toy, val=False)
    generate_relevant(toy.transactions, directory_toy, val=True)


def main():
    tree = datasets.HMDatasetDirectoryTree()
    dataset = datasets.HMDataset(tree=tree, folds="threesets")

    save_full_dataset_as_parquet(dataset, "full")

    # generate toys and their relevant datasets
    generate_toy_and_relevant(dataset, "toy", 10000)
    generate_toy_and_relevant(dataset, "toy_1k", 1000)
    generate_toy_and_relevant(dataset, "toy500", 500)


if __name__ == "__main__":
    main()
