import numpy as np

from hmcollab import directories
from hmcollab import datasets
from hmcollab import splitter


def main():
    tree = datasets.HMDatasetDirectoryTree()
    dataset = datasets.HMDataset(tree=tree, folds="threesets")

    r = np.random.RandomState(42)

    fivehundred = r.choice(
        dataset.transactions_y.customer_id.unique(), size=500, replace=False
    )

    portion = splitter.CustomerPortion(fivehundred)
    pruned_dataset = portion.split(dataset)

    pruned_dataset.customers.to_csv(
        directories.testdata("fivehundred/customers.csv"), index=False
    )
    pruned_dataset.articles.to_csv(
        directories.testdata("fivehundred/articles.csv"), index=False
    )
    pruned_dataset.transactions.to_csv(
        directories.testdata("fivehundred/transactions_train.csv"), index=False
    )


if __name__ == "__main__":
    main()
