from scipy.interpolate import interp1d

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt


def sync_behavior_data_to_photometry(
    behavior_data: pd.DataFrame, photometry_data: pd.DataFrame
) -> pd.DataFrame:
    b_timestamps = behavior_data["Timestamp"]
    p_timestamps = photometry_data["Timestamp"]

    shared_timestamps = p_timestamps[
        (b_timestamps.min() <= p_timestamps) & (p_timestamps <= b_timestamps.max())
    ]

    # linearly interpolate continuous variables onto photometry clock
    interpolation_map = {
        "WheelPosition": "linear",
        "RunningSpeed": "linear",
        "Reward": "nearest",
        "Gain": "nearest",
        "TrialNumber": "nearest",
    }
    old_cols = ["mouse_id", "training_group", "session_date"]

    new_cols = [shared_timestamps]
    for col, kind in interpolation_map.items():
        interpolate = interp1d(b_timestamps, behavior_data.loc[:, col], kind=kind)
        new_col = interpolate(shared_timestamps)
        new_cols.append(list(new_col))

    new_col_names = ["Timestamp"] + list(interpolation_map.keys())
    synced_behavior_data = pd.DataFrame(dict(zip(new_col_names, new_cols)))

    for col in old_cols:
        synced_behavior_data[col] = behavior_data[col].unique()[0]
        synced_behavior_data[col] = synced_behavior_data[col].astype("category")

    print(
        f"Synced behavior to photometry for "
        f"{behavior_data['training_group'].unique()[0]}, "
        f"{behavior_data['mouse_id'].unique()[0]}, "
        f"{behavior_data['session_date'].unique()[0]}\n"
    )

    return synced_behavior_data


def get_gain_change_events(behavior_data: pd.DataFrame) -> pd.DataFrame:
    # gain change events for photometry data
    gain = np.asarray(behavior_data["Gain"])
    position = np.asarray(behavior_data["WheelPosition"])
    trial_num = np.asarray(behavior_data["TrialNumber"])

    # Find gain changes
    timestamp = np.where(np.diff(trial_num) > 0)[0]

    # Get information about gain changes
    magnitude = [np.diff(gain)[i] for i in timestamp]
    gain_before = gain[timestamp]
    gain_after = gain[timestamp + 1]
    sinusoid_frequency = 50 / 3  # from Romain
    delta_position = np.mod(np.diff(position)[timestamp], sinusoid_frequency)
    phase_shift = delta_position / sinusoid_frequency * 360

    gain_change_mapping = {
        2.7: "++",
        2.0: "+",
        0.7: "+",
        0: "0",
        -0.7: "-",
        -2.0: "-",
        -2.7: "--",
    }

    return pd.DataFrame(
        {
            "Timestamp": timestamp,
            "magnitude": magnitude,
            "gain_change": [gain_change_mapping[i] for i in magnitude],
            "delta_position": delta_position,
            "phase_shift": phase_shift,
            "gain_before": gain_before,
            "gain_after": gain_after,
        }
    )


def extract_data_of_interest(
    gain_change_events: pd.DataFrame,
    photometry_data: pd.DataFrame,
    window: tuple = (30, 80),
) -> pd.DataFrame:
    # get the columns we'd like to have in the df
    gain_change_cols = list(gain_change_events.columns)[1:]

    # initialize empty list of dfs to concatenate
    dfs = []

    for i, timestamp in enumerate(gain_change_events["Timestamp"]):
        # extract the window around the timestamp
        df = photometry_data.loc[timestamp - window[0] : timestamp + window[1]]

        # mark the moment gain changes as 0
        df["Timestamp"] -= list(df.loc[timestamp - 1 : timestamp + 4, "Timestamp"])[0]

        for col in gain_change_cols:
            df[col] = gain_change_events.loc[i, col]

        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def align_photometry_data(photometry_data, gain_change_events, window_samples):
    aligned_data = []
    for idx, gain_change_event in gain_change_events.iterrows():
        gain_change_sample = gain_change_event["Timestamp"]
        start_sample = gain_change_sample - window_samples[0]
        end_sample = gain_change_sample + window_samples[1]

        aligned_photometry = photometry_data[
            (photometry_data["Timestamp"] >= start_sample)
            & (photometry_data["Timestamp"] < end_sample)
        ].copy()

        aligned_photometry["gain_before"] = gain_change_event["gain_before"]
        aligned_photometry["gain_after"] = gain_change_event["gain_after"]

        aligned_data.append(aligned_photometry)

    return pd.concat(aligned_data)


def plot_gain_change_traces(aligned_data, window_samples, savepath=None):
    xticks_sample = range(window_samples[0], window_samples[1], 100)
    xticks_sec = [f"{(x / 1000):.1f}" for x in xticks_sample]

    g = sns.FacetGrid(
        aligned_data,
        col="gain_before",
        row="gain_after",
        sharex=True,
        sharey=True,
        aspect=1.5,
    )
    g.map(sns.lineplot, "Timestamp", "df_over_f", alpha=0.5, lw=1)
    g.set_titles("Gain before {col_name} | Gain after {row_name}")
    g.set_axis_labels("Time (s)", "dF/F")
    g.set(xticks=xticks_sample)
    g.set_xticklabels(xticks_sec)
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle("Aligned Gain Change Traces")

    if savepath:
        plt.savefig(savepath)
        plt.close()
    else:
        plt.show()
