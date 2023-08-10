import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

DATA_PATH = "./mouse_data/"

###########################################################
# Old data with Megha's 2 mice
###########################################################

data = pd.read_csv(DATA_PATH + "final_data_mouse_1_RL1.csv")

# rename gain changes
data["Gain Change Magnitude"] = data["Gain Change Magnitude"].apply(lambda x: x[3:])

# experiment params
sinusoid_frequency = 50 / 3
glitch_limits = (sinusoid_frequency * 0.2, sinusoid_frequency * 0.8)

# define position glitch (0 +/- 20% : no, rest: yes)
data["Phase shift"] = "other"
data.loc[
    (data["Delta Position value"] <= glitch_limits[0])
    | (data["Delta Position value"] >= glitch_limits[1]),
    "Phase shift",
] = "Close to 0"
data.loc[
    (data["Delta Position value"] >= glitch_limits[0])
    & (data["Delta Position value"] <= glitch_limits[1]),
    "Phase shift",
] = "Close to pi"


# split trials by overall glitch magnitude
data["Glitch magnitude"] = "minor"
data.loc[
    (data["Phase shift"] == "Close to pi") & (data["Gain Change Magnitude"] != "0"),
    "Glitch magnitude",
] = "major"
data.loc[
    (data["Phase shift"] == "Close to 0") & (data["Gain Change Magnitude"] == "0"),
    "Glitch magnitude",
] = "none"

# split trials by glitch type
data["Glitch type"] = "None"
data.loc[
    (data["Phase shift"] == "Close to pi") & (data["Gain Change Magnitude"] == "0"),
    "Glitch type",
] = "Position glitch"
data.loc[
    (data["Phase shift"] == "Close to 0") & (data["Gain Change Magnitude"] != "0"),
    "Glitch type",
] = "Speed glitch"
data.loc[
    (data["Phase shift"] == "Close to pi") & (data["Gain Change Magnitude"] != "0"),
    "Glitch type",
] = "Both"

####################################################
# Plots
####################################################

zero_mark = 100 / 3
xticks = [50 / 3, 100 / 3, 50, 200 / 3]
xtick_labels = [-0.5, 0, 0.5, 1]


def format_axes(ax: plt.Axes) -> None:
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels)
    yl = ax.get_ylim()
    ax.vlines(zero_mark, yl[0], yl[1], ls="--", color="lightgrey")
    ax.set_ylim(yl)


fig, axs = plt.subplots(1, 4, figsize=(24, 6))

# Effect of overall surprise
g = sns.lineplot(
    data=data,
    x="Time Aligned to Gain Change (sec)",
    y="dF/F",
    hue="Glitch magnitude",
    style="Glitch magnitude",
    palette="flare",
    ax=axs[0],
).set_title("Effect of overall surprise")
format_axes(g.axes)

# Modality specific vs combined glitches
g = sns.lineplot(
    data=data,
    x="Time Aligned to Gain Change (sec)",
    y="dF/F",
    hue="Glitch type",
    style="Glitch type",
    palette=["darkorchid", "royalblue", "indianred", "gray"],
    ax=axs[1],
    hue_order=["Both", "Speed glitch", "Position glitch", "None"],
).set_title("Effect of glitch type")
format_axes(g.axes)

# Signal split by gain change (no Phase shift)
colors = sns.color_palette("Spectral", 5)
colors[2] = "grey"
g = sns.lineplot(
    data=data[data["Phase shift"] == "Close to 0"],
    x="Time Aligned to Gain Change (sec)",
    y="dF/F",
    hue="Gain Change Magnitude",
    palette=colors,
    style="Glitch magnitude",
    ax=axs[2],
)
handles, labels = g.get_legend_handles_labels()
g.get_legend().remove()
g.legend(handles=handles[:6], labels=labels[:6])
g.set_title("Effect of gain change only")
format_axes(g.axes)

# Signal split by delta position (no gain change)
g = sns.lineplot(
    data=data[
        (data["Gain Change Magnitude"] == "0") & (data["Phase shift"] != "other")
    ],
    x="Time Aligned to Gain Change (sec)",
    y="dF/F",
    hue="Phase shift",
    style="Phase shift",
    palette=["orangered", "gray"],
    ax=axs[3],
).set_title("Effect of position glitch only")
format_axes(g.axes)

plt.tight_layout()
plt.savefig("Figure.png")
plt.close()


# Adiitional plots
# split by delta position: NEED TRIAL NUM & SESSION NUM
g = sns.lineplot(
    data=data,
    x="Time Aligned to Gain Change (sec)",
    y="dF/F",
    hue="Delta Position",
    style="Delta Position",
    hue_order=[
        "0.00 - 3.35",
        "3.35 - 6.71",
        "6.71 - 10.06",
        "10.06 - 13.41",
        "13.41 - 16.77",
    ],
    palette="coolwarm",
).set_title("Split by magnitude of phase shift (all gain changes)")
format_axes(g.axes)
plt.savefig("split_by_phase_shift.png")
plt.close()


##############################################
# New data from Josh & Jaimie
##############################################

# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd

# DATA_PATH = "./mouse_data/"

data_path = DATA_PATH + "data_new/"

# example file
file = data_path + "JS4_04_10_23_000000.csv"

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


dfs = []
for file in csv_files:
    print(f"Processing file {file}")
    data = pd.read_csv(data_path + file)
    dfs.append(add_id_cols(pivot_data(data), file))

df = pd.concat(dfs, ignore_index=True)
