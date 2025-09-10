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
from csep.core.forecasts import CatalogForecast
forecast = CatalogForecast.from_file('synthetic_catalog.csv')
# %%
import csep
from csep.core import regions
from csep.utils import time_utils

start = time_utils.strptime_to_utc_datetime('2025-01-01 00:00:00')
end = time_utils.strptime_to_utc_datetime('2025-12-31 23:59:59')
obs_catalog = csep.query_comcat(start, end)
# %%
from csep.core.evaluations.catalog_tests import number_test, magnitude_test, spatial_test, pseudolikelihood_test

# number_test 示例
num_res = number_test(forecast, obs_catalog)
mag_res = magnitude_test(forecast, obs_catalog)
spatial_res = spatial_test(forecast, obs_catalog)
pseudo_res = pseudolikelihood_test(forecast, obs_catalog)
