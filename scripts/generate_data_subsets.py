import numpy as np

from hmcollab import directories
from hmcollab import datasets
from hmcollab import splitter


def customer_split(dataset, customer_count):
    r = np.random.RandomState(42)

    selected_custumers = r.choice(
        dataset.transactions_y.customer_id.unique(), size=customer_count, replace=False
    )

    portion = splitter.CustomerPortion(selected_custumers)
    return portion.split(dataset)


def save_main_data(pruned_dataset, base_path):
    customers_fn = directories.qualifyname(base_path, "customers.csv")
    article_fn = directories.qualifyname(base_path, "articles.csv")
    transaction_fn = directories.qualifyname(base_path, "transactions_train.csv")
    pruned_dataset.customers.to_csv(customers_fn, index=False)
    pruned_dataset.articles.to_csv(article_fn, index=False)
    pruned_dataset.transactions.to_csv(transaction_fn, index=False)


def generate_toy(dataset):
    """
    Creating the toy dataset (also train and test)
    Generate toy with 10k random customer
    """
    pruned_dataset = customer_split(dataset, 10000)
    save_main_data(pruned_dataset, directories.data("toy"))


def generate_test_data(dataset):
    pruned_dataset = customer_split(dataset, 500)
    save_main_data(pruned_dataset, directories.testdata("fivehundred"))


def main():
    tree = datasets.HMDatasetDirectoryTree()
    dataset = datasets.HMDataset(tree=tree, folds="threesets", toy=False)

    generate_test_data(dataset)
    generate_toy(dataset)


if __name__ == "__main__":
    main()
