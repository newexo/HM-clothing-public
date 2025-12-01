import numpy as np
import os
from collections import OrderedDict

from hmcollab import directories
from hmcollab import datasets
from hmcollab import splitter


def split_transactions_by_month(transactions_df):
    """
    Given a DataFrame with a datetime64 column `t_dat`,
    return an OrderedDict mapping (year, month) → monthly_dataframe.
    """

    # Extract year and month
    df = transactions_df.copy()
    df["year"] = df.t_dat.dt.year
    df["month"] = df.t_dat.dt.month

    # Group and collect
    monthly = OrderedDict()
    for (y, m), group in df.groupby(["year", "month"]):
        monthly[(y, m)] = group.drop(columns=["year", "month"])

    return monthly


def customer_split(dataset, customer_count):
    r = np.random.RandomState(42)

    selected_customers = r.choice(
        dataset.transactions_y.customer_id.unique(), size=customer_count, replace=False
    )

    portion = splitter.CustomerPortion(selected_customers)
    return portion.split(dataset)


def save_parquet_and_csv(df, parquet_path):
    """
    Given a dataframe and a .parquet filename,
    write both the parquet file and a CSV file with the same basename.
    """
    # Write parquet
    df.to_parquet(parquet_path, index=False)

    # Construct the CSV filename next to the parquet file
    csv_path = parquet_path.replace(".parquet", ".csv")
    df.to_csv(csv_path, index=False)


def save_main_data(pruned_dataset, base_path):
    customers_fn = directories.qualifyname(base_path, "customers.parquet")
    article_fn = directories.qualifyname(base_path, "articles.parquet")
    transaction_fn = directories.qualifyname(base_path, "transactions_train.parquet")

    save_parquet_and_csv(pruned_dataset.customers, customers_fn)
    save_parquet_and_csv(pruned_dataset.articles, article_fn)
    save_parquet_and_csv(pruned_dataset.transactions, transaction_fn)


def save_full_dataset_as_parquet(dataset, dir_name="full"):
    """
    Save the full (unpruned) dataset as parquet + csv files.
    Additionally, split the transactions by year and month and
    save each shard under full/transactions/YYYY/MM/.
    """
    path = directories.data(dir_name)
    if not os.path.exists(path):
        os.mkdir(path)

    customers_fn = directories.qualifyname(path, "customers.parquet")
    articles_fn = directories.qualifyname(path, "articles.parquet")
    transactions_fn = directories.qualifyname(path, "transactions_train.parquet")

    # Save dimension tables (full)
    save_parquet_and_csv(dataset.customers, customers_fn)
    save_parquet_and_csv(dataset.articles, articles_fn)

    # Combine X + Y transactions
    full_transactions = dataset.transactions_x.append(
        dataset.transactions_y, ignore_index=True
    )

    # Save unified full transactions file
    save_parquet_and_csv(full_transactions, transactions_fn)

    # split into monthly partitions and save those as well
    monthly = split_transactions_by_month(full_transactions)

    # Base directory for monthly shards: full/transactions/
    tx_base = os.path.join(path, "transactions")
    if not os.path.exists(tx_base):
        os.mkdir(tx_base)

    for (year, month), df_month in monthly.items():
        # Directory: full/transactions/YYYY/
        year_dir = os.path.join(tx_base, f"{year}")
        if not os.path.exists(year_dir):
            os.mkdir(year_dir)

        # Directory: full/transactions/YYYY/MM/
        month_dir = os.path.join(year_dir, f"{month:02d}")
        if not os.path.exists(month_dir):
            os.mkdir(month_dir)

        # File name: YYYY-MM.parquet
        fn = directories.qualifyname(month_dir, f"{year}-{month:02d}.parquet")
        save_parquet_and_csv(df_month, fn)

    print(f"Saved full dataset to: {path}")


def save_relevant_data(relevant_data, base_path, val=False):
    name = "relevant.parquet" if not val else "relevant_val.parquet"
    relevant_fn = directories.qualifyname(base_path, name)

    save_parquet_and_csv(relevant_data, relevant_fn)


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

    path = directories.data(dir_name)
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

    generate_toy_and_relevant(dataset, "toy", 10000)
    generate_toy_and_relevant(dataset, "toy_1k", 1000)
    generate_toy_and_relevant(dataset, "toy500", 500)


if __name__ == "__main__":
    main()
