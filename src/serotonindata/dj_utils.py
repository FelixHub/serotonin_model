import datetime
import os
from os.path import join as pjoin

from foundation_tables import Date, Mouse

DATA_PATH = "./mouse_data/"


def update_date(data_path: str = DATA_PATH) -> None:
    """
    Add dates when we had some data to the date table
    """
    for mouse in Mouse().fetch(as_dict=True):
        mouse_path = pjoin(data_path, mouse["prefix"], str(mouse["mouse_id"]))
        for datestr in os.listdir(mouse_path):
            session_date = datetime.date(
                *[int(x) for x in [datestr[:4], datestr[4:6], datestr[6:]]]
            )
            Date().insert1({"session_date": session_date}, skip_duplicates=True)
