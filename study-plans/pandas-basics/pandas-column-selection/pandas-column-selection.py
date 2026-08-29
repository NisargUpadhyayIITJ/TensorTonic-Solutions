import pandas as pd

def select_column(data, column):
    """
    Returns: dict with 'values' (list) and 'length' (int)
    """
    df = pd.DataFrame(data)
    result = {
        "values" : list(df[column].values),
        "length" : int(df.shape[0])
    }
    return result