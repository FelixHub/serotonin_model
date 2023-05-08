from handlers import BehaviorDataHandler, PhotometryDataHandler
from helpers import (
    sync_behavior_data_to_photometry,
    get_gain_change_events,
    extract_data_of_interest,
)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class DataProcessor:
    def __init__(
        self,
        behavior_handler: BehaviorDataHandler,
        photometry_handler: PhotometryDataHandler,
    ):
        self.behavior_handler = behavior_handler
        self.photometry_handler = photometry_handler

    def process_all_sessions(self):
        all_data = pd.DataFrame()

        for behavior_file, photometry_file in zip(
            self.behavior_handler.csv_files, self.photometry_handler.csv_files
        ):
            print(behavior_file, photometry_file)

            if "RL2" in behavior_file or "20220116" in behavior_file:
                print("Skipping this one.")
            else:
                behavior_data = self.behavior_handler.process(behavior_file)
                photometry_data = self.photometry_handler.process(photometry_file)

                synced_behavior_data = sync_behavior_data_to_photometry(
                    behavior_data, photometry_data
                )
                gain_change_events = get_gain_change_events(synced_behavior_data)

                df_plot = extract_data_of_interest(gain_change_events, photometry_data)

                all_data = pd.concat([all_data, df_plot], ignore_index=True)

        return all_data


if __name__ == "__main__":
    # instantiate data compilers
    handlers = BehaviorDataHandler(), PhotometryDataHandler()

    # Instantiate DataProcessor and process all sessions
    data_processor = DataProcessor(*handlers)
    all_data = data_processor.process_all_sessions()

    # Plot the results
    sns.lineplot(
        data=all_data, x="Timestamp", y="df_over_f", hue="gain_change", n_boot=300
    )
    plt.show()
