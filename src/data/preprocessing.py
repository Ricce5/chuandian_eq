import logging
import pandas as pd
import numpy as np
import os
from pathlib import Path
import src.features.seismic_features as sf
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

def process_cn_catalog(dat_file):
    column_names = ['Year', 'Month', 'Day', 'Hour', 'Minute', 'Second', 'Latitude', 'Longitude', 'Depth', 'Magnitude']
    df = pd.read_csv(dat_file, header=None, names=column_names, sep=r"\s+")
    logger.debug("Loaded CN catalog preview:\n%s", df.head())
    data_input = df.to_numpy()
    year = data_input[:, 0].astype(int)
    month = data_input[:, 1].astype(int)
    day = data_input[:, 2].astype(int)
    hour = data_input[:, 3].astype(int)
    minute = data_input[:, 4].astype(int)
    second = data_input[:, 5]

    data = np.column_stack((year, month, day, hour, minute, second))
    jd0 = sf.cal2jd([1970, 1, 1, 0, 0, 0])
    jd = np.array([sf.cal2jd(d) - jd0 for d in data])

    df.loc[:, 't'] = jd
    df['dt'] = df['t'].diff().fillna(0)
    df = df[['t', 'Magnitude', 'Latitude', 'Longitude', 'Depth', 'dt']]
    csv_file = os.path.splitext(dat_file)[0] + '.csv'
    df.to_csv(csv_file, index=False)
    return df




def process_synthetic_catlog(dat_file):
    column_names =["ID1", "ID2", "t", "Magnitude", "Depth", "Longitude", "Latitude"]
    df = pd.read_csv(dat_file, header=None, names=column_names, sep=r"\s+")
    df['dt'] = df['t'].diff().fillna(0)
    df = df[['t', 'Magnitude', 'Latitude', 'Longitude', 'Depth', 'dt']]
    csv_file = os.path.splitext(dat_file)[0] + '.csv'
    df.to_csv(csv_file, index=False)
    
    return df

def load_and_filter_catalog(base_dir, Mc):
    raw_dir = os.path.join(base_dir, 'raw')
    csv_files = [f for f in os.listdir(raw_dir) if f.endswith('.csv')]
    logger.info("CSV files found: %s", csv_files)

    if len(csv_files) == 1:
        chosen_file = csv_files[0]
    elif len(csv_files) > 1:
        processed_files = [f for f in csv_files if f.startswith('processed_')]
        if len(processed_files) != 1:
            raise ValueError(f"Expected exactly one 'processed_' CSV file among multiple files, found {len(processed_files)}: {processed_files}")
        chosen_file = processed_files[0]
        logger.info("Multiple CSV files found. Using processed file: %s", chosen_file)
    else:
        raise ValueError("No CSV files found in the 'raw' directory.")
    
    df = pd.read_csv(os.path.join(raw_dir, chosen_file))
    df = df[df['Magnitude'] >= Mc].reset_index(drop=True)
    df = df.sort_values(by='t').reset_index(drop=True)
    df['dt_unfiltered'] = df['dt']
    df['dt'] = df['t'].diff().fillna(0)
    return df


def _resolve_recast_csv_path(csv_file):
    candidate = Path(csv_file)
    if candidate.exists():
        return candidate

    raw_dir = candidate.parent
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory not found for CSV lookup: {raw_dir}")

    csv_candidates = sorted(
        path
        for path in raw_dir.glob("*.csv")
        if not path.name.lower().startswith("processed_")
    )
    if not csv_candidates:
        raise FileNotFoundError(f"No CSV files found in directory: {raw_dir}")

    target_stem = candidate.stem.lower()
    preferred_matches = [
        path
        for path in csv_candidates
        if path.stem.lower() == target_stem and "catalog" not in path.stem.lower()
    ]
    if preferred_matches:
        return preferred_matches[0]

    stem_matches = [path for path in csv_candidates if path.stem.lower() == target_stem]
    if stem_matches:
        return stem_matches[0]

    non_catalog = [path for path in csv_candidates if "catalog" not in path.stem.lower()]
    if len(non_catalog) == 1:
        return non_catalog[0]

    if len(csv_candidates) == 1:
        return csv_candidates[0]

    display = [path.name for path in csv_candidates]
    raise FileNotFoundError(
        f"Could not resolve recast CSV for expected path '{candidate}'. "
        f"Available CSV files in '{raw_dir}': {display}"
    )


def _resolve_recast_metadata_path(metadata_file, csv_path):
    metadata_path = Path(metadata_file)
    if metadata_path.exists():
        return metadata_path

    dataset_dir = csv_path.parent.parent
    catalogs_dir = dataset_dir / "catalogs"
    if catalogs_dir.exists():
        hashed_metadata = sorted(
            path
            for path in catalogs_dir.glob("*/metadata.pt")
            if path.is_file()
        )
        if len(hashed_metadata) == 1:
            return hashed_metadata[0]
        if len(hashed_metadata) > 1:
            raise FileNotFoundError(
                f"Multiple metadata.pt files found under '{catalogs_dir}'; "
                "please specify metadata_file explicitly. "
                f"Found: {[str(path) for path in hashed_metadata]}"
            )

    raise FileNotFoundError(
        f"Metadata file not found: {metadata_path}. "
        "Also searched for hashed metadata under "
        f"'{catalogs_dir}'."
    )


def _resolve_recast_time_column(df):
    for col in ("time", "date_time", "datetime", "ts"):
        if col in df.columns:
            return col
    raise ValueError(
        "Could not find a timestamp column for recast catalog. "
        "Expected one of: ['time', 'date_time', 'datetime', 'ts']"
    )


def _infer_start_ts_from_df(df):
    time_col = _resolve_recast_time_column(df)
    ts = pd.to_datetime(df[time_col], errors="coerce")
    valid_ts = ts.dropna()
    if valid_ts.empty:
        raise ValueError(
            f"Could not parse timestamps from column '{time_col}' to infer start_ts."
        )
    return valid_ts.min().floor("D")


def process_recast_catalog(csv_file, metadata_file):
    import torch

    resolved_csv = _resolve_recast_csv_path(csv_file)
    df = pd.read_csv(resolved_csv)

    try:
        resolved_metadata = _resolve_recast_metadata_path(metadata_file, resolved_csv)
    except FileNotFoundError as exc:
        start_ts = _infer_start_ts_from_df(df)
        logger.warning(
            "Metadata not found for '%s'; inferred start_ts=%s from catalog timestamps. Details: %s",
            resolved_csv,
            start_ts,
            exc,
        )
    else:
        meta_data = torch.load(resolved_metadata, weights_only=False)
        start_ts = pd.to_datetime(meta_data['start_ts'])

    time_col = _resolve_recast_time_column(df)
    df['ts'] = pd.to_datetime(df[time_col], errors="coerce")
    if df['ts'].isna().all():
        raise ValueError(f"All timestamps are NaT after parsing column '{time_col}'.")

    jd0 = sf.cal2jd(start_ts)
    valid_ts = df['ts'].dropna()
    jd = np.full(len(df), np.nan, dtype=float)
    jd[valid_ts.index] = np.array([sf.cal2jd(d) - jd0 for d in valid_ts])
    df['t'] = jd
    df['dt'] = df['t'].diff().fillna(0)
    df = df[['t', 'magnitude', 'latitude', 'longitude', 'depth', 'dt', 'ts']]
    save_file = resolved_csv.parent / f'processed_{resolved_csv.stem}.csv'
    df.rename(columns={'magnitude': 'Magnitude', 'latitude': 'Latitude', 'longitude': 'Longitude', 'depth': 'Depth'}, inplace=True)
    df.to_csv(save_file, index=False)
    return df

def calculate_catalog_statistics(df):
    stats = {
        'tau_unfiltered': float(df['dt_unfiltered'].mean()),
        'tau_q025': float(df['dt'].quantile(0.25)),
        'tau_q05': float(df['dt'].quantile(0.5)),
        'tau_mean': float(df['dt'].mean()),
        'tau_max': float(df['dt'].max()),
        'tau_min': float(df['dt'].min()),
        'time_max': float(df['t'].max()),
        'time_mean': float(df['t'].mean()),
        'mag_completeness': float(df['Magnitude'].min()),
        'mag_mean': float(df['Magnitude'].mean()),
        'mag_max': float(df['Magnitude'].max()),
        'mag_min': float(df['Magnitude'].min()),
    }

    # b-value (maximum likelihood) calculation
    mags = df['Magnitude'].dropna().to_numpy()
    Mc = stats['mag_completeness']
    b_val = float('nan')
    b_std = float('nan')
    b_n = 0

    if mags.size > 0:
        # estimate bin width dM from unique magnitudes if possible, fallback to 0.1
        unique_mags = np.unique(np.round(mags, 6))
        if unique_mags.size > 1:
            diffs = np.diff(unique_mags)
            nonzero = diffs[diffs > 1e-8]
            dM = float(nonzero.min()) if nonzero.size > 0 else 0.1
        else:
            dM = 0.1

        # threshold correction (Mc - dM/2)
        threshold = Mc - dM / 2.0
        mags_above = mags[mags >= Mc]  # use Mc as completeness cutoff
        b_n = mags_above.size

        if b_n >= 2:
            mean_excess = mags_above.mean() - threshold
            if mean_excess > 0:
                b_val = np.log10(np.e) / mean_excess  # log10(e) / mean(M - threshold)
                b_std = b_val / np.sqrt(b_n)  # approximate standard error (Aki, 1965)

    stats.update({
        'b_value': None if np.isnan(b_val) else float(b_val),
        'b_std':   None if np.isnan(b_std) else float(b_std),
        'b_n':     int(b_n),
    })

    logger.info("Catalog statistics: %s", stats)
    return stats


def plot_dt_distributions(dfs, names=None, bins=100, figsize=(20, 4)):
    if names is None:
        names = [f'df{i+1}' for i in range(len(dfs))]

    plt.figure(figsize=figsize)
    
    for i, (df, name) in enumerate(zip(dfs, names)):
        plt.subplot(1, len(dfs), i + 1)
        dt = df['dt'].dropna()
        plt.hist(dt, bins=bins, alpha=0.7, color=f'C{i}')
        mean = dt.mean()
        std = dt.std()

        plt.axvline(mean, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean:.2f}')
        plt.axvline(mean + std, color='green', linestyle='dotted', linewidth=1, label=f'Std: {std:.2f}')
        plt.axvline(mean - std, color='green', linestyle='dotted', linewidth=1)
        plt.title(f'{name} dt distribution')
        plt.xlabel('dt')
        plt.ylabel('Count')
        plt.legend()

    plt.tight_layout()
    plt.show()

def estimate_mc_max_curvature(mags, bin_width=0.1, plot=True):
    mags = np.asarray(mags)
    mags = mags[~np.isnan(mags)]  # remove NaN values
    
    if len(mags) == 0:
        raise ValueError("Magnitude array is empty, cannot estimate Mc")
    m_min = np.floor(mags.min() * 10) / 10.0
    m_max = np.ceil(mags.max() * 10) / 10.0
    
    bins = np.arange(m_min, m_max + bin_width, bin_width)
    counts, edges = np.histogram(mags, bins=bins)
    bin_centers = (edges[:-1] + edges[1:]) / 2.0

    if len(counts) == 0:
        raise ValueError("Magnitude range too narrow to compute histogram")
    idx_max = np.argmax(counts)
    mc = bin_centers[idx_max]

    if plot:
        plt.figure(figsize=(6, 4))
        plt.bar(bin_centers, counts, width=bin_width, align="center", edgecolor="k")
        plt.axvline(mc, linestyle="--", linewidth=2, label=f"Mc = {mc:.2f}")
        plt.xlabel("Magnitude")
        plt.ylabel("Count")
        plt.title("Magnitude Frequency Histogram (Max Curvature Method)")
        plt.legend()
        plt.tight_layout()
        plt.show()
    return mc, bin_centers, counts

def plot_magnitude_distribution(df_or_mags, bin_width=0.1, mc=None, figsize=(6, 4), ax=None, save_path=None, show=True):
    if isinstance(df_or_mags, pd.DataFrame):
        if 'Magnitude' not in df_or_mags.columns:
            raise ValueError("DataFrame must contain a 'Magnitude' column.")
        mags = df_or_mags['Magnitude'].dropna().to_numpy()
    else:
        mags = np.asarray(df_or_mags)
        mags = mags[~np.isnan(mags)]

    if mags.size == 0:
        raise ValueError("Magnitude array is empty, cannot plot distribution.")

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True
    else:
        fig = ax.figure

    m_min = np.floor(mags.min() / bin_width) * bin_width
    m_max = np.ceil(mags.max() / bin_width) * bin_width
    bins = np.arange(m_min, m_max + bin_width, bin_width)

    ax.hist(mags, bins=bins, color='C0', alpha=0.75, edgecolor='white', linewidth=0.6)
    ax.set_xlabel('Magnitude')
    ax.set_ylabel('Count')
    ax.set_title('Magnitude Distribution')

    mean_mag = float(np.mean(mags))
    ax.axvline(mean_mag, color='C3', linestyle='--', linewidth=1.5, label=f'Mean: {mean_mag:.2f}')

    if mc is not None:
        ax.axvline(mc, color='C2', linestyle='-', linewidth=1.5, label=f'Mc: {mc:.2f}')

    ax.legend()
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
    if show:
        plt.show()
    elif created_fig:
        plt.close(fig)

    return fig, ax
