import numpy as np

def df_to_X_y(df, window_size=5, horizon=5):
  df_as_np = np.array(df)

  X = []
  y = []

  for i in range(len(df_as_np) - window_size - horizon):
    row = df_as_np[i:i + window_size]
    X.append(row)

    labels = df_as_np[i + window_size:i + window_size + horizon]
    y.append(labels)

  return np.array(X), np.array(y)