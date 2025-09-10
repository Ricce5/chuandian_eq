import pandas as pd
import csep
from csep.core.forecasts import CatalogForecast
from csep.core.evaluations.catalog_tests import number_test, magnitude_test, spatial_test, pseudolikelihood_test
from csep.utils import time_utils

# 1. 生成或者加载观测目录
start = time_utils.strptime_to_utc_datetime('2025-01-01 00:00:00')
end = time_utils.strptime_to_utc_datetime('2025-12-31 23:59:59')
obs_catalog = csep.query_comcat(start, end)

# 2. 构造简单模拟目录
df = pd.DataFrame({
    'LON': [...],
    'LAT': [...],
    'MAG': [...],
    'ORIGIN_TIME': [...],
    'DEPTH': [...],
    'CATALOG_ID': [...],
    'EVENT_ID': [...],
})
df.to_csv('synthetic_catalog.csv', index=False)
forecast = CatalogForecast.from_file('synthetic_catalog.csv')

# 3. 进行评估测试
results = {
    'number_test': number_test(forecast, obs_catalog),
    'magnitude_test': magnitude_test(forecast, obs_catalog),
    'spatial_test': spatial_test(forecast, obs_catalog),
    'pseudolikelihood_test': pseudolikelihood_test(forecast, obs_catalog),
}
print(results)
