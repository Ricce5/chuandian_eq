# %%
import csep
from csep.core import poisson_evaluations as poisson
from csep.utils import datasets, time_utils, plots

start = time_utils.strptime_to_utc_datetime('2006-11-12 00:00:00.0')
end   = time_utils.strptime_to_utc_datetime('2011-11-12 00:00:00.0')

forecast = csep.load_gridded_forecast(
    datasets.helmstetter_aftershock_fname,
    start_date=start, end_date=end,
    name='helmstetter_aftershock'
)

catalog = csep.query_comcat(forecast.start_time, forecast.end_time, min_magnitude=forecast.min_magnitude)

catalog = catalog.filter_spatial(forecast.region)

s_res = poisson.spatial_test(forecast, catalog)


n_res = poisson.number_test(forecast, catalog)

ax = plots.plot_poisson_consistency_test(s_res, plot_args={'xlabel': 'Spatial likelihood'})


csep.write_json(s_res, 'spatial_test.json')
csep.write_json(n_res, 'number_test.json')
print('Done.')

# %%
