def find_missing_channel(df):
    return df[df["acquisition_channel"].isna()]
