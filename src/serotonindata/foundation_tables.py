import os
from os.path import join as pjoin

import datajoint as dj
import pandas as pd
from dj_connection import connect_noTLS

DATA_PATH = "./mouse_data/"

connect_noTLS()
schema = dj.schema("rlmp_gainChange", locals())


@schema
class Mouse(dj.Manual):
    definition = """
    # Mice
    prefix: varchar(5)             # mouse training group
    mouse_id: smallint             # unique mouse id
    ---
    dob = NULL: date                      # mouse date of birth
    sex = NULL: enum('M', 'F', 'U')       # sex of mouse - Male, Female, or Unknown/Unclassified
    """


@schema
class Date(dj.Manual):
    definition = """
    # Dates when we had some data collection
    session_date: date            # date where we had some form of data collection
    ---
    """


@schema
class BehaviorSession(dj.Imported):
    definition = """
    # Behavior sessions
    -> Mouse
    -> Date
    beha_session_time: varchar(20)      # time of session
    ---
    beha_path: varchar(128)        # path to behavior data
    beha_data: longblob            # matrix of behavior data, flexible to task
    """
    key_source = Mouse() * Date()

    def make(self, key):
        date_path = pjoin(
            DATA_PATH,
            key["prefix"],
            str(key["mouse_id"]),
            key["session_date"].strftime("%Y%m%d"),
        )

        if not (self & key):
            if os.path.isdir(date_path):
                for file in os.listdir(date_path):
                    if file.lower().startswith("behavior"):
                        beha_path = pjoin(date_path, file)
                        session_time = beha_path[
                            beha_path.index("T") + 1 : beha_path.index(".csv")
                        ]
                        beha_data = pd.read_csv(beha_path).to_numpy()

                        key.update(
                            beha_session_time=session_time,
                            beha_path=beha_path,
                            beha_data=beha_data,
                        )
                        self.insert1(key)
                        print(
                            f"Processed new behavior session from {key['prefix']} {key['mouse_id']} {key['session_date']} \n"
                        )


@schema
class PhotometrySession(dj.Imported):
    definition = """
    # Photometry session
    -> Mouse
    -> Date
    photometry_session_time: varchar(20)             # time of recording
    ---
    photometry_path: varchar(128)         # path to photometry data
    """
    key_source = Mouse() * Date()

    def make(self, key):
        date_path = pjoin(
            DATA_PATH,
            key["prefix"],
            str(key["mouse_id"]),
            key["session_date"].strftime("%Y%m%d"),
        )

        if not (self & key):
            if os.path.isdir(date_path):
                for file in os.listdir(date_path):
                    if file.lower().startswith("photometry"):
                        photometry_path = pjoin(date_path, file)
                        session_time = photometry_path[
                            photometry_path.index("T")
                            + 1 : photometry_path.index(".csv")
                        ]

                        key.update(
                            photometry_session_time=session_time,
                            photometry_path=photometry_path,
                        )
                        self.insert1(key)
                        print(
                            f"Processed new photometry session from {key['prefix']} {key['mouse_id']} {key['session_date']} \n"
                        )
