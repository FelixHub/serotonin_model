from abc import ABC, abstractmethod, abstractproperty
from glob import glob
from scipy.signal import butter, filtfilt, medfilt
from scipy.stats import linregress

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


DATA_PATH = "./mouse_data/"


class DataHandler(ABC):
    def __init__(self, data_path: str = DATA_PATH):
        self.data_path = data_path
        csv_files = glob(f"{data_path}/*/*/*/*.csv", recursive=True)
        self.csv_files = [file for file in csv_files if self.is_file_type_correct(file)]

    @abstractproperty
    def file_type(self) -> str:
        pass

    def is_file_type_correct(self, file):
        return self.file_type in file.lower()

    @staticmethod
    def get_mouse_id(file) -> int:
        return int(file.split("/")[-3])

    @staticmethod
    def get_training_group(file) -> str:
        return file.split("/")[-4]

    @staticmethod
    def get_session_date(file):
        return file.split("/")[-2]

    def append_metadata_columns(self, df: pd.DataFrame, file: str) -> pd.DataFrame:
        df["mouse_id"] = self.get_mouse_id(file)
        df["training_group"] = self.get_training_group(file)
        df["session_date"] = self.get_session_date(file)
        return df

    @abstractmethod
    def process(self, file) -> pd.DataFrame:
        pass


class BehaviorDataHandler(DataHandler):
    def __init__(self, data_path: str = DATA_PATH):
        super().__init__(data_path)
        self.columns = {
            "RL1": [
                "Timestamp",
                "WheelPosition",
                "RunningSpeed",
                "Reward",
                "Gain",
                "TrialNumber",
            ],
            "RL2": [
                "Reward",
                "Gain",
                "TrialNumber",
                "Timestamp",
                "WheelPosition",
                "RunningSpeed",
            ],
        }

    @property
    def file_type(self) -> str:
        return "behavior"

    def process(self, file) -> pd.DataFrame:
        col_names = self.columns[self.get_training_group(file)]
        df = pd.read_csv(file, names=col_names, header=None)
        return self.append_metadata_columns(df, file)


class PhotometryDataHandler(DataHandler):
    def __init__(self, data_path: str = DATA_PATH, processing_params: dict = None):
        super().__init__(data_path)

        if not processing_params:
            self.processing_params = {
                "photometry_preprocessing_param_id": 0,
                "median_filt_kernel_size": 5,
                "lowpass_filter_frequency": 10,
                "photobleaching_estim_method": self.get_baseline,
                "photobleaching_estim_params": {"polyfit_deg": 4},
                "baseline_fluorescence_signal": "denoised",
                "baseline_fluorescence_lowpass_freq": 0.001,
            }

    @property
    def file_type(self) -> str:
        return "photometry"

    def process(self, file) -> pd.DataFrame:
        return (
            pd.read_csv(file)
            .pipe(self._extract_dF_over_F, file)
            .pipe(self.append_metadata_columns, file)
        )

    @staticmethod
    def get_baseline(timestamps, signal_denoised, polyfit_deg=4):
        """Return baseline signal by polynomial fit"""
        coefs = np.polyfit(timestamps, signal_denoised, deg=polyfit_deg)
        signal_baseline = np.polyval(coefs, timestamps)
        signal_bleachCorrected = signal_denoised - signal_baseline
        return signal_baseline, signal_bleachCorrected

    def _extract_dF_over_F(self, df: pd.DataFrame, file) -> pd.DataFrame:
        """
        Extract dF/F from raw photometry signal after performing
        denoising, baseline correction, and motion correction.
        """

        # photometry_folder = file.rsplit("/", 1)[0]

        # get rid of initial artifact
        aftifact_frames = 6
        df = df[aftifact_frames:]

        # unpack raw data
        sample_wavelength = df["Flags"]
        roi_g = df["Region1G"]
        roi_r = df["Region0R"]
        sample_times_all = df["Timestamp"]
        sample_rate = 1 / np.median(np.diff(sample_times_all))

        # define channel masks
        channel_415_mask = sample_wavelength == 17
        channel_470_mask = sample_wavelength == 18

        # extract timestamps and raw signal for each channel mask
        sample_times_415 = sample_times_all[channel_415_mask]
        sample_times_470 = sample_times_all[channel_470_mask]
        raw_signal_415 = roi_g[channel_415_mask]
        raw_signal_470 = roi_g[channel_470_mask]

        # linear interpolation to get everyone on 470 timeframe
        raw_signal_415 = np.interp(sample_times_470, sample_times_415, raw_signal_415)

        # # plot raw signals
        # plt.figure()
        # plt.plot(sample_times_470, raw_signal_415, color="r", label="415 signal")
        # plt.plot(sample_times_470, raw_signal_470, color="g", label="470 signal")
        # plt.legend()
        # plt.savefig((photometry_folder + "/raw_signal_new.png"))
        # plt.close()

        # Median filtering to remove electrical artifacts
        denoised_415_signal = medfilt(
            raw_signal_415,
            kernel_size=self.processing_params["median_filt_kernel_size"],
        )
        denoised_470_signal = medfilt(
            raw_signal_470,
            kernel_size=self.processing_params["median_filt_kernel_size"],
        )

        # Lowpass filter - zero phase filtering to avoid distorting
        b, a = butter(
            2,
            self.processing_params["lowpass_filter_frequency"],
            btype="low",
            fs=sample_rate,
        )
        denoised_415_signal = filtfilt(b, a, denoised_415_signal)
        denoised_470_signal = filtfilt(b, a, denoised_470_signal)

        # photobleaching baseline estimation
        baseline_func = self.processing_params["photobleaching_estim_method"]
        g415_baseline, g415_bleachCorrected = baseline_func(
            sample_times_470,
            denoised_415_signal,
            **self.processing_params["photobleaching_estim_params"],
        )
        g470_baseline, g470_bleachCorrected = baseline_func(
            sample_times_470,
            denoised_470_signal,
            **self.processing_params["photobleaching_estim_params"],
        )

        # Motion correction
        # get linear regression betw g415 and g470 signal
        slope, intercept, _, _, _ = linregress(
            x=g415_bleachCorrected, y=g470_bleachCorrected
        )
        GCaMP_est_motion = intercept + slope * g415_bleachCorrected
        GCaMP_motionCorrected = g470_bleachCorrected - GCaMP_est_motion

        # get baseline for dF/F
        b, a = butter(
            2,
            self.processing_params["baseline_fluorescence_lowpass_freq"],
            btype="low",
            fs=sample_rate,
        )

        if self.processing_params["baseline_fluorescence_signal"] == "denoised":
            baseline_fluorescence = filtfilt(b, a, denoised_470_signal, padtype="even")
        elif (
            self.processing_params["baseline_fluorescence_signal"] == "motion_corrected"
        ):
            baseline_fluorescence = filtfilt(
                b, a, GCaMP_motionCorrected, padtype="even"
            )

        # calculate dF/F
        GCaMP_dF_F = GCaMP_motionCorrected / baseline_fluorescence

        # # visualize final dF/F
        # fig, ax = plt.subplots(1, 2)
        # ax[0].plot(sample_times_470, GCaMP_dF_F * 100, "g")
        # ax[0].set_xlabel("Time (seconds)")
        # ax[0].set_ylabel("GCaMP delta-F/F (%)")
        # ax[0].set_title("GCaMP dF/F")
        # ax[1].plot(sample_times_470, GCaMP_dF_F * 100, "g")
        # ax[1].set_xlabel("Time (seconds)")
        # ax[1].set_ylabel("GCaMP delta-F/F (%)")
        # ax[1].set_title("GCaMP dF/F First 100 seconds")
        # ax[1].set_xlim(sample_times_470[0], sample_times_470[0] + 100)
        # plt.savefig(photometry_folder + "/final_dF_over_F_new.png")
        # plt.close()

        df_processed = pd.DataFrame(columns=["Timestamp", "df_over_f"])
        df_processed["Timestamp"] = sample_times_470
        df_processed["df_over_f"] = GCaMP_dF_F

        return df_processed
