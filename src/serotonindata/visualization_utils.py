from typing import List, Tuple

import numpy as np
from behavior_processing import GainChangeEvents
from photometry_preprocessing import ProcessedPhotometry


def get_gain_change_data(
    key: dict, pooling: str, time_pre: float, time_post: float
) -> Tuple[List, List, List, List, np.ndarray, List, int, int, float]:
    """
    Gather photometry data around gain change
    """

    def align_gain_change_psth(
        df_over_f: np.ndarray,
        gain_change_samples: List[int],
        samples_pre: int,
        samples_post: int,
    ) -> np.ndarray:
        gain_change_psth = []
        n_samples = len(df_over_f)

        for sample in gain_change_samples:
            trial_trace = df_over_f[(sample - samples_pre) : (sample + samples_post)]

            if sample + samples_post - n_samples > 0:
                padded_trace = np.concatenate(
                    (trial_trace, np.full(sample + samples_post - n_samples, np.nan))
                )
                gain_change_psth.append(padded_trace[:, np.newaxis])
            else:
                gain_change_psth.append(trial_trace[:, np.newaxis])

        return np.concatenate(gain_change_psth, axis=1).T

    def fetch_gain_change_event_data(key: dict) -> Tuple:
        return (GainChangeEvents() & key).fetch1(
            "gain_change_samples",
            "gain_change_magnitudes",
            "delta_position",
            "gain_pre",
            "gain_post",
        )

    def fetch_processed_photometry_data(key: dict) -> Tuple:
        return (ProcessedPhotometry() & key).fetch1("sample_times", "df_over_f")

    if pooling == "none":
        (
            gain_change_samples,
            gain_change_magnitudes,
            delta_position,
            gain_pre,
            gain_post,
        ) = fetch_gain_change_event_data(key)

        sample_times, df_over_f = fetch_processed_photometry_data(key)

        n_samples = len(sample_times)
        dt = np.median(np.diff(sample_times)).round(2)
        samples_pre = np.round(time_pre / dt).astype(int)
        samples_post = np.round(time_post / dt).astype(int)

        gain_change_psth = align_gain_change_psth(
            df_over_f, gain_change_samples, samples_pre, samples_post
        )
        session_number = []

    elif np.isin(pooling, ["mouse", "all"]):
        session_keys = (
            (GainChangeEvents() & key).fetch("KEY")
            if pooling == "mouse"
            else (GainChangeEvents()).fetch("KEY")
        )

        example_sample_times = (ProcessedPhotometry() & session_keys[0]).fetch1(
            "sample_times"
        )
        dt = np.median(np.diff(example_sample_times)).round(2)
        samples_pre = np.round(time_pre / dt).astype(int)
        samples_post = np.round(time_post / dt).astype(int)
        print("dt",dt)

        gain_change_mapping = {
            2.7: "0: ++",
            2.0: "1: +",
            0.7: "1: +",
            0: "2: 0",
            -0.7: "3: -",
            -2.0: "3: -",
            -2.7: "4: --",
        }

        (
            gain_change_magnitudes,
            delta_position,
            gain_pre,
            gain_post,
            gain_change_psth,
            session_number,
        ) = ([], [], [], [], [], [])

        for i_session, session_key in enumerate(session_keys):
            (
                s_gain_change_samples,
                s_gain_change_magnitudes,
                s_delta_position,
                s_gain_pre,
                s_gain_post,
            ) = fetch_gain_change_event_data(session_key)
            df_over_f = (ProcessedPhotometry & session_key).fetch1("df_over_f")
            s_gain_change_psth = align_gain_change_psth(
                df_over_f, s_gain_change_samples, samples_pre, samples_post
            )

            gain_change_magnitudes.extend(
                [gain_change_mapping[i] for i in s_gain_change_magnitudes]
            )
            delta_position.extend(s_delta_position)
            gain_pre.extend(s_gain_pre)
            gain_post.extend(s_gain_post)
            gain_change_psth.append(s_gain_change_psth)
            session_number.extend([i_session] * len(s_gain_change_magnitudes))

        gain_change_magnitudes = np.array(gain_change_magnitudes)
        delta_position = np.array(delta_position)
        gain_pre = np.array(gain_pre)
        gain_post = np.array(gain_post)
        gain_change_psth = np.concatenate(gain_change_psth, axis=0)
        session_number = np.array(session_number)

    return (
        gain_change_magnitudes,
        delta_position,
        gain_pre,
        gain_post,
        gain_change_psth,
        session_number,
        samples_pre,
        samples_post,
        dt,
    )
