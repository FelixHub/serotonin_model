import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from visualization_utils import get_gain_change_data


class TraceVisualizer:
    def __init__(
        self,
        key,
        time_pre=1,
        time_post=1.5,
        n_boot=300,
        xticks_sec=[-0.5, 0, 0.5, 1],
        n_delta_position_bins=5,
        pooling="all",
    ):
        self.key = key
        self.time_pre = time_pre
        self.time_post = time_post
        self.n_boot = n_boot
        self.xticks_sec = xticks_sec
        self.n_delta_position_bins = n_delta_position_bins
        self.pooling = pooling
        self.gain_change_df = self.process_data()

        # helper variables
        self.xticks_sample = None
        self.samples_pre = None

    def process_data(self):
        # Gather gain change information and photometry data
        (
            gain_change_magnitudes,
            delta_position,
            gain_pre,
            gain_post,
            gain_change_psth,
            session_number,
            samples_pre,
            samples_post,
            dt,
        ) = get_gain_change_data(self.key, self.pooling, self.time_pre, self.time_post)

        # process analysis settings
        self.xticks_sample = [
            np.argmin(
                np.abs(
                    np.arange(samples_pre + samples_post) * dt
                    - self.time_pre
                    - this_xtick
                )
            )
            for this_xtick in self.xticks_sec
        ]
        self.samples_pre = samples_pre

        # bin delta position around max potential delta
        sinusoid_frequency = 50 / 3  # from romain
        delta_position_binEdges = np.linspace(
            0, sinusoid_frequency + 0.1, self.n_delta_position_bins + 1
        )
        delta_position_digitized = (
            np.digitize(delta_position, delta_position_binEdges) - 1
        )
        bin_names = [
            "%.2f - %.2f" % tuple(delta_position_binEdges[[i, i + 1]])
            for i in range(len(delta_position_binEdges) - 1)
        ]
        delta_position_discr = np.full(len(delta_position_digitized), "              ")
        for i_bin in range(self.n_delta_position_bins):
            delta_position_discr[delta_position_digitized == i_bin] = np.full(
                len(delta_position_discr[delta_position_digitized == i_bin]),
                bin_names[i_bin],
            )
        delta_position_discr = pd.Categorical(
            delta_position_discr, categories=bin_names, ordered=True
        )

        # pandas dataframe it
        gain_change_df = pd.DataFrame(gain_change_psth)
        gain_change_df["Gain Change Magnitude"] = gain_change_magnitudes
        gain_change_df["Delta Position value"] = delta_position
        gain_change_df["Delta Position"] = delta_position_discr
        gain_change_df["Gain Pre"] = gain_pre
        gain_change_df["Gain Post"] = gain_post

        gain_change_df = gain_change_df.melt(
            id_vars=[
                "Gain Change Magnitude",
                "Delta Position",
                "Delta Position value",
                "Gain Pre",
                "Gain Post",
            ]
        ).rename(
            columns={"variable": "Time Aligned to Gain Change (sec)", "value": "dF/F"}
        )
        gain_change_df.sort_values("Gain Change Magnitude", inplace=True)

        return gain_change_df

    def plot_gain_change_magnitude(self, ax):
        sns.lineplot(
            x="Time Aligned to Gain Change (sec)",
            y="dF/F",
            hue="Gain Change Magnitude",
            data=self.gain_change_df,
            palette="RdBu",
            ax=ax,
            n_boot=self.n_boot,
        )

    def plot_delta_position(self, ax):
        sns.lineplot(
            x="Time Aligned to Gain Change (sec)",
            y="dF/F",
            hue="Delta Position",
            data=self.gain_change_df,
            palette="RdBu",
            ax=ax,
            n_boot=self.n_boot,
        )

    def plot_no_gain_change(self, ax):
        sns.lineplot(
            x="Time Aligned to Gain Change (sec)",
            y="dF/F",
            hue="Delta Position",
            data=self.gain_change_df[
                self.gain_change_df["Gain Change Magnitude"] == "2: 0"
            ],
            palette="RdBu",
            ax=ax,
            n_boot=self.n_boot,
        ).set_title("Only trials with no gain change")

    def plot_no_phase_shift(self, ax):
        sns.lineplot(
            x="Time Aligned to Gain Change (sec)",
            y="dF/F",
            hue="Gain Change Magnitude",
            data=self.gain_change_df.loc[
                (self.gain_change_df["Delta Position value"] <= 3)
                | (self.gain_change_df["Delta Position value"] >= 13.5)
            ],
            palette="RdBu",
            ax=ax,
            n_boot=self.n_boot,
        ).set_title("Only trials with no phase shift")

    def format_axes(self, axs: list[plt.Axes]):
        for i_ax in range(4):
            axs[i_ax].set_xticks(self.xticks_sample)
            axs[i_ax].set_xticklabels(self.xticks_sec)
            yl = axs[i_ax].get_ylim()
            axs[i_ax].vlines(self.samples_pre, yl[0], yl[1], ls="--", color="k")
            axs[i_ax].set_ylim(yl)

    def format_figure(self, fig: plt.figure):
        if self.pooling == "none":
            self.key["session_date_str"] = self.key["session_date"].strftime("%D")
            fig.suptitle(
                "{prefix} {mouse_id} {session_date_str}".format(**self.key), fontsize=16
            )
        elif self.pooling == "mouse":
            fig.suptitle("{prefix} {mouse_id}".format(**self.key), fontsize=16)
        elif self.pooling == "all":
            fig.suptitle("All mice", fontsize=16)
        fig.tight_layout()

    def save_plot(self, savepath):
        if self.pooling == "none":
            self.key["session_date_savestr"] = self.key["session_date"].strftime(
                "%m%d%Y"
            )
            folder = "session_gain_change_traces"
            filename = "{prefix}_{mouse_id}_{session_date_savestr}.png".format(
                **self.key
            )
        elif self.pooling == "mouse":
            folder = "mouse_gain_change_traces"
            filename = "{prefix}_{mouse_id}.png".format(**self.key)
        elif self.pooling == "all":
            folder = "pooled_gain_change_traces"
            filename = "pooled_gain_change_traces.png"
        else:
            raise NotImplementedError

        full_path = os.path.join(savepath, folder)
        os.makedirs(full_path, exist_ok=True)
        plt.savefig(os.path.join(full_path, filename))
        plt.close()

    def visualize(self, savepath=None):
        self.process_data()
        fig, axs = plt.subplots(1, 4, figsize=(20, 7), sharey=True, sharex=True)
        self.plot_gain_change_magnitude(axs[0])
        self.plot_delta_position(axs[1])
        self.plot_no_gain_change(axs[2])
        self.plot_no_phase_shift(axs[3])

        # Additional formatting and saving code
        self.format_axes(axs)
        self.format_figure(fig)
        if savepath:
            self.save_plot(savepath)


def trace_visualization(
    key,
    time_pre=1,
    time_post=1.5,
    n_boot=300,
    xticks_sec=[-0.5, 0, 0.5, 1],
    n_delta_position_bins=5,
    pooling="none",
    savepath=None,
):
    trace_visualizer = TraceVisualizer(
        key, time_pre, time_post, n_boot, xticks_sec, n_delta_position_bins, pooling
    )
    trace_visualizer.visualize(savepath)
