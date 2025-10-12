# %%
# Most of the core functionality can be imported from the top-level csep package. 
import csep
# Or you could import directly from submodules, like csep.core or csep.utils submodules.
from csep.core import regions, catalog_evaluations
from csep.core import poisson_evaluations as poisson
from csep.utils import datasets, time_utils, comcat, plots
import numpy
import cartopy
# %%
### Set up model parameters

# Start and end time
start_time = time_utils.strptime_to_utc_datetime("1992-06-28 11:57:34.14")
end_time = time_utils.strptime_to_utc_datetime("1992-07-28 11:57:34.14")

# Magnitude bins properties
min_mw = 4.95
max_mw = 8.95
dmw = 0.1

# Create space and magnitude regions. The forecast is already filtered in space and magnitude
magnitudes = regions.magnitude_bins(min_mw, max_mw, dmw)
region = regions.california_relm_region()

# Bind region information to the forecast (this will be used for binning of the catalogs)

space_magnitude_region = regions.create_space_magnitude_region(region, magnitudes)
# %%
forecast = csep.load_catalog_forecast('ucerf3-landers.csv',
                                      start_time = start_time,
                                      end_time = end_time,
                                      region = space_magnitude_region)
# %%
