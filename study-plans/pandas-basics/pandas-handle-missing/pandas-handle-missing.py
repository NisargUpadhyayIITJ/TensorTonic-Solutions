import pandas as pd

def handle_missing(data, fill_value):
    """
    Returns: dict with 'null_counts' (dict) and 'cleaned_data' (dict)
    """
    df = pd.DataFrame(data)
    new_df = df.fillna(fill_value)
    return {
        "null_counts" : dict(df.isna().sum()),
        "cleaned_data" : new_df.to_dict("list")
    }