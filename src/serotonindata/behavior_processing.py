import datajoint as dj
import numpy as np
from foundation_tables import BehaviorSession, PhotometrySession
from photometry_preprocessing import ProcessedPhotometry
from scipy.interpolate import interp1d

schema = dj.schema("rlmp_gainChange", locals())


@schema
class PhotometrySyncBehavior(dj.Computed):
    definition = """
      # Behavior time series on photometry clock
      -> PhotometrySession
      -> BehaviorSession
      ---
      wheel_position : longblob
      running_speed : longblob 
      reward : longblob 
      gain : longblob 
      trial_num : longblob 
      """
    # note some stuff going on in key_source: need to ignore session_time b/c we
    # start recording photometry and beh at diff times
    # and we don't care about making new syncBeha acr diff preprocessing steps,
    # so leave this out as well
    key_source = PhotometrySession() * BehaviorSession()

    def make(self, key):
        # Load photometry sample times
        sample_times = (ProcessedPhotometry() & key).fetch1("sample_times")

        # load behavior data
        beha_data = (BehaviorSession() & key).fetch1("beha_data")
        beha_file_header = [
            "Timestamp",
            "WheelPosition",
            "RunningSpeed",
            "Reward",
            "Gain",
            "TrialNumber",
        ]
        behavior_sample_times = beha_data[:, 0]
        shared_sample_times = sample_times[
            (sample_times >= behavior_sample_times[0])
            & (sample_times <= behavior_sample_times[-1])
        ]
        samples_pre = len(np.where(sample_times < behavior_sample_times[0])[0])
        samples_post = len(np.where(sample_times > behavior_sample_times[-1])[0])

        # linearly interpolate continuous variables onto photometry clock
        linear_interp_fns = [
            interp1d(behavior_sample_times, beha_data[:, i_variable], kind="linear")
            for i_variable in range(1, 3)
        ]
        wheel_position, running_speed = [
            linear_interp_fns[i_variable](shared_sample_times)
            for i_variable in range(2)
        ]
        # nearest neighbor interpolate categorical variables onto photometry clock
        nearest_interp_fns = [
            interp1d(behavior_sample_times, beha_data[:, i_variable], kind="nearest")
            for i_variable in range(3, 6)
        ]
        reward, gain, trial_num = [
            nearest_interp_fns[i_variable](shared_sample_times)
            for i_variable in range(3)
        ]
        gain[0] = np.nan

        # pad behavior data w/ nans
        wheel_position, running_speed, reward, gain, trial_num = [
            np.concatenate(
                (np.nan + np.zeros(samples_pre), x, np.nan + np.zeros(samples_post))
            )
            for x in [wheel_position, running_speed, reward, gain, trial_num]
        ]

        # insert
        key["wheel_position"] = wheel_position
        key["running_speed"] = running_speed
        key["reward"] = reward
        key["gain"] = gain
        key["trial_num"] = trial_num
        self.insert1(key)

        print(
            "Synced behavior to photometry for {prefix} {mouse_id} {session_date} \n".format(
                **key
            )
        )


@schema
class GainChangeEvents(dj.Computed):
    definition = """ 
    # Sample numbers of gain change events and information about them
    -> PhotometrySyncBehavior
    ---
    gain_change_samples : blob 
    gain_change_magnitudes : blob
    delta_position : blob
    gain_pre : blob 
    gain_post : blob 
    """
    key_source = PhotometrySyncBehavior()

    def make(self, key):
        # load sync behavior data
        gain, position, trial_num = (PhotometrySyncBehavior() & key).fetch1(
            "gain", "wheel_position", "trial_num"
        )
        print("gain",gain)
        # Find gain changes
        # gain_changes = np.unique(np.diff(gain[~np.isnan(gain)]))
        # gain_changes = gain_changes[np.abs(gain_changes) >= 0]
        # gain_change_samples = np.where(np.isin(np.diff(gain), gain_changes))[0]
        gain_change_samples = np.where(np.diff(trial_num) > 0)[0]
        print("gain_change_samples",gain_change_samples)
        # Get information about gain changes
        gain_change_magnitudes = [np.diff(gain)[ix] for ix in gain_change_samples]
        gain_pre = gain[gain_change_samples]
        gain_post = gain[gain_change_samples + 1]
        print(gain_pre)
        print(gain_post)
        sinusoid_frequency = 50 / 3  # from Romain
        delta_position = np.mod(
            np.diff(position)[gain_change_samples], sinusoid_frequency
        )

        # Package and insert
        key["gain_change_samples"] = gain_change_samples
        key["gain_change_magnitudes"] = gain_change_magnitudes
        key["delta_position"] = delta_position
        key["gain_pre"] = gain_pre
        key["gain_post"] = gain_post
        self.insert1(key)
