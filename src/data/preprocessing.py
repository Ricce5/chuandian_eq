import pandas as pd
import numpy as np
import os
import src.features.seismic_features as sf
import matplotlib.pyplot as plt

def process_cn_catalog(dat_file):
    column_names = ['Year', 'Month', 'Day', 'Hour', 'Minute', 'Second', 'Latitude', 'Longitude', 'Depth', 'Magnitude']
    df = pd.read_csv(dat_file, header=None, names=column_names, sep='\s+')
    print(df)
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
    df = pd.read_csv(dat_file, header=None, names=column_names, sep='\s+')
    df['dt'] = df['t'].diff().fillna(0)
    df = df[['t', 'Magnitude', 'Latitude', 'Longitude', 'Depth', 'dt']]
    csv_file = os.path.splitext(dat_file)[0] + '.csv'
    df.to_csv(csv_file, index=False)
    
    return df

def load_and_filter_catalog(base_dir, Mc):
    raw_dir = os.path.join(base_dir, 'raw')
    csv_files = [f for f in os.listdir(raw_dir) if f.endswith('.csv')]
    print("CSV files found:", csv_files)

    if len(csv_files) == 1:
        chosen_file = csv_files[0]
    elif len(csv_files) > 1:
        processed_files = [f for f in csv_files if f.startswith('processed_')]
        if len(processed_files) != 1:
            raise ValueError(f"Expected exactly one 'processed_' CSV file among multiple files, found {len(processed_files)}: {processed_files}")
        chosen_file = processed_files[0]
        print(f"Multiple CSV files found. Using processed file: {chosen_file}")
    else:
        raise ValueError("No CSV files found in the 'raw' directory.")
    
    df = pd.read_csv(os.path.join(raw_dir, chosen_file))
    df = df[df['Magnitude'] >= Mc].reset_index(drop=True)
    df = df.sort_values(by='t').reset_index(drop=True)
    df['dt_unfiltered'] = df['dt']
    df['dt'] = df['t'].diff().fillna(0)
    return df


def process_recast_catalog(csv_file, metadata_file):
    import torch
    from pathlib import Path
    df = pd.read_csv(csv_file)
    meta_data = torch.load(metadata_file, weights_only=False)
    start_ts = pd.to_datetime(meta_data['start_ts'])
    df['ts'] = pd.to_datetime(df['time'])
    jd0 = sf.cal2jd(start_ts)
    jd = np.array([sf.cal2jd(d) - jd0 for d in df['ts']])
    df['t'] = jd
    df['dt'] = df['t'].diff().fillna(0)
    df = df[['t', 'magnitude', 'latitude', 'longitude', 'depth', 'dt', 'ts']]
    csv_file = Path(csv_file)
    save_file = csv_file.parent / f'processed_{csv_file.stem}.csv'
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
        'mag_min': float(df['Magnitude'].min())
    }
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