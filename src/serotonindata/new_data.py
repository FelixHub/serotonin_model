
##############################################
# New data from Josh & Jaimie
##############################################

# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd

# DATA_PATH = "./mouse_data/"

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

DATA_PATH = "./"
data_path = DATA_PATH + "data_new/"

# example file
file = data_path + "JA4331_05_02_23_000000.csv"

data = pd.read_csv(file)


import os

csv_files = [file for file in os.listdir(data_path) if file.endswith(".csv")]


def get_prefix(file: str) -> str:
    return file[:2]


def get_mouse_id(file: str) -> str:
    return file.split("_")[0][2:]


def get_session_date(file: str) -> str:
    day, month, year = file.split("_")[1:4]
    return "20" + year + month + day


def get_session_id(file: str) -> str:
    return file.split("_")[-1].split(".")[0]


def add_id_cols(df: pd.DataFrame, file: str) -> pd.DataFrame:
    df["mouse_id"] = get_mouse_id(file)
    df["session_id"] = get_session_id(file)
    df["session_date"] = get_session_date(file)
    df["prefix"] = get_prefix(file)
    df["uid"] = file.split(".")[0]
    return df


def map_delta_gain_column(x: int) -> str:
    mapping = {
        8: "++",
        6: "+",
        2: "+",
        0: "0",
        -2: "-",
        -6: "-",
        -8: "--",
    }
    return mapping[x]


def pivot_data(data: pd.DataFrame) -> pd.DataFrame:
    gain_change_df = data.melt(
        id_vars=[
            "prev_track_length",
            "delta_gain",
            "glitch_degrees",
            "Unnamed: 0",
        ]
    ).rename(
        columns={
            "variable": "Time Aligned to Gain Change (sec)",
            "value": "dF/F",
            "Unnamed: 0": "Trial number",
            "glitch_degrees": "Phase shift",
            "delta_gain": "Gain Change Magnitude",
        }
    )
    gain_change_df["Gain Change Magnitude"] = gain_change_df[
        "Gain Change Magnitude"
    ].apply(map_delta_gain_column)
    return gain_change_df

'''
dfs = []
for file in csv_files:
    print(f"Processing file {file}")
    data = pd.read_csv(data_path + file)
    dfs.append(add_id_cols(pivot_data(data), file))

df = pd.concat(dfs, ignore_index=True)
data = dfs[0]
'''


data_df = add_id_cols(pivot_data(data), file)

time_samples = data_df['Time Aligned to Gain Change (sec)'].unique().astype(float)
sample_rate = (np.median(np.diff(time_samples)))
print("sampling rate",1/sample_rate)


# we want to write smart function to compute the gain pre and gain post from these mice
data[data['delta_gain'] == 8]


zero_mark = 100 / 3
xticks = [50 / 3, 100 / 3, 50, 200 / 3]
xtick_labels = [-0.5, 0, 0.5, 1]


fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Effect of overall surprise
g = sns.lineplot(
    data=data_df,
    x="Time Aligned to Gain Change (sec)",
    y="dF/F",
    hue="Phase shift",
    style="Phase shift",
    palette="flare",
    ax=axs[0],
).set_title("Effect of overall surprise")


plt.show()
