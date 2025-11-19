import numpy as np
from pyproj import Geod

def GlobRectGrid(resol, lonbdn, latbnd):
    """
    Python version of GlobRectGrid_ver3 (MATLAB).
    
    Parameters
    ----------
    resol : [dLon, dLat]  resolution in degrees
    lonbdn : [lon_min, lon_max]  longitude boundaries
    latbnd : [lat_min, lat_max]  latitude boundaries
    
    Returns
    -------
    LonGMat : 2D array of grid center longitudes
    LatGMat : 2D array of grid center latitudes
    areamat : 2D array of grid cell areas (km^2)
    grid_vec_format : N×3 array [lon, lat, area]
    """

    # === 1. Generate lon/lat grid boundaries ===
    longrid = np.arange(lonbdn[0], lonbdn[1] + resol[0], resol[0])
    latgrid = np.arange(latbnd[0], latbnd[1] + resol[1], resol[1])

    # === 2. Compute grid centers ===
    loncen = (longrid[:-1] + longrid[1:]) / 2
    latcen = (latgrid[:-1] + latgrid[1:]) / 2

    # === 3. Create meshgrid ===
    LonGMat, LatGMat = np.meshgrid(loncen, latcen)

    # === 4. Area calculation using WGS84 ellipsoid ===
    geod = Geod(ellps='WGS84')

    # Preallocate
    area = np.zeros(LonGMat.size)

    # Grid half-size
    dlon = resol[0] / 2
    dlat = resol[1] / 2

    lon_flat = LonGMat.ravel()
    lat_flat = LatGMat.ravel()

    # === Compute area for each grid cell ===
    for i in range(len(area)):
        lonC = lon_flat[i]
        latC = lat_flat[i]

        # Four corners: clockwise
        lons = [lonC - dlon, lonC + dlon, lonC + dlon, lonC - dlon]
        lats = [latC - dlat, latC - dlat, latC + dlat, latC + dlat]

        poly_area, _ = geod.polygon_area_perimeter(lons, lats)
        area[i] = abs(poly_area) / 1e6   # convert m^2 → km^2

    # Reshape to matrix
    areamat = area.reshape(LonGMat.shape)

    # Vector output
    grid_vec_format = np.vstack([LonGMat.ravel(), LatGMat.ravel(), area]).T

    return LonGMat, LatGMat, areamat, grid_vec_format
