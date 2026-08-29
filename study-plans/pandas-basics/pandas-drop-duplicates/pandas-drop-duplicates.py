import pandas as pd

def drop_duplicates(data):
    """
    Returns: list [rows_before, rows_after, cleaned_data]
    """
    df = pd.DataFrame(data)
    new_df = df.drop_duplicates()
    return [df.shape[0], new_df.shape[0], new_df.to_dict("list")]
