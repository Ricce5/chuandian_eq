# %%
import numpy as np
from src.utils.catalog_tests import magnitude_test_from_counts
# --- Construct a fake Catalog class ---
class FakeCatalog:
    def __init__(self, counts):
        self._counts = np.array(counts)

    def magnitude_counts(self):
        return self._counts

# --- Construct observed data (obs) ---
observed_catalog = FakeCatalog([5, 10, 15, 8, 2])   # Simulated observed magnitude distribution

# --- Construct forecast data ---
forecast_catalogs = [
    FakeCatalog([4, 9, 16, 7, 3]),
    FakeCatalog([6, 11, 14, 9, 1]),
    FakeCatalog([5, 10, 15, 8, 2])  # Exactly equal to the observed data
]

# --- Call the test function ---
result_dict = magnitude_test_from_counts(
    forecast_catalogs,
    observed_catalog,
    verbose=True,
    return_quantiles=True
)

# --- Output results ---
print("Test distribution:", result_dict["test_distribution"])
print("Observed d-statistic:", result_dict["obs_d_statistic"])
print("Scaled union histogram:", result_dict["scaled_union_histogram"])
print("Quantiles (delta_1, delta_2):", result_dict["delta_1"], result_dict["delta_2"])

# %%
from src.utils.catalog_tests import run_magnitude_test_result
run_magnitude_test_result(result_dict, plot=True)
# %%
