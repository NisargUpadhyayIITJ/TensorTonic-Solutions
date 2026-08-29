import pandas as pd

def create_dataframe(data):
    """
    Returns: dict with 'data', 'shape', 'columns'
    """
    df = pd.DataFrame(data)
    result = {
        "data": df.to_dict("list"),
        "shape": list(df.shape),
        "columns": df.columns.tolist(),
    }
    return result
    
    