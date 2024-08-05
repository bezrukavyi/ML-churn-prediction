import numpy as np


def drop_high_correlation(dataframe):
    threshold = 0.95

    corr_matrix = dataframe.corr().abs()

    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    print(f"Features to drop: {to_drop}")

    # Видалення ознак
    dataframe.drop(columns=to_drop, inplace=True)

    return dataframe
