# %%
import pandas as pd

df = pd.DataFrame({
    'LON': [34.05, 34.10],
    'LAT': [-118.25, -118.20],
    'MAG': [4.5, 4.8],
    'ORIGIN_TIME': ['2025-01-01T00:00:00', '2025-01-02T12:00:00'],
    'DEPTH': [10.0, 8.0],
    'CATALOG_ID': [0, 0],
    'EVENT_ID': [0, 1],
})
df.to_csv('synthetic_catalog.csv', index=False)
# %%