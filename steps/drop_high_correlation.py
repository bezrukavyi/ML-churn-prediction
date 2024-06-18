def drop_high_correlation(dataframe):
    # Create correlation matrix
    corr_matrix = dataframe.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Find index of feature columns with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

    print(f"Features to drop: {to_drop}")

    # Drop features
    dataframe.drop(dataframe[to_drop], axis=1, inplace=True)

    return dataframe
