#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/9/6 15:52
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   train.py
# @Desc     :   

from pandas import DataFrame, concat
from sklearn.model_selection import train_test_split
from streamlit import (empty, sidebar, subheader, session_state, slider,
                       caption, number_input, button, rerun)

from utils.helper import Timer

empty_messages: empty = empty()
empty_train_title: empty = empty()
empty_train_table: empty = empty()
empty_test_title: empty = empty()
empty_test_table: empty = empty()

all_sessions: list[str] = ["data", "timer_train", "X", "Y"]
for all_session in all_sessions:
    session_state.setdefault(all_session, None)
page_sessions: list[str] = ["x_train", "x_test", "y_train", "y_test"]
for page_session in page_sessions:
    session_state.setdefault(page_session, None)

with sidebar:
    if session_state["data"] is None:
        empty_messages.error("Please upload a dataset in the Home page.")
    else:
        empty_messages.info("The dataset is ready for model training.")
        subheader("Model Training Settings")

        seed: int = number_input(
            "Random Seed",
            min_value=0, max_value=10000, value=9527, step=1,
            help="The random seed for reproducibility.",
        )

        size: int = slider(
            "Test Set Size (%)",
            min_value=10, max_value=30, value=30, step=5,
            help="The size of the test set.",
        )
        caption(f"The test set size is **{size} %**.")

        if session_state["x_train"] is None:
            empty_messages.warning("Please split the dataset first.")
            if button("Split the Dataset", type="primary", width="stretch"):
                with Timer("Split the dataset") as t:
                    # Split the data into training and testing sets
                    x_train, x_test, y_train, y_test = train_test_split(
                        session_state.X, session_state.Y,
                        test_size=size / 100,
                        random_state=seed,
                        shuffle=True,
                        stratify=None
                    )
                    session_state["x_train"], session_state["x_test"] = DataFrame(x_train), DataFrame(x_test)
                    session_state["y_train"], session_state["y_test"] = DataFrame(y_train), DataFrame(y_test)
                session_state.timer_train = t
                rerun()
        else:
            empty_messages.info(f"{session_state.timer_train} Dataset split successfully.")

            empty_train_title.markdown(f"Training Set {len(session_state.x_train)} samples")
            empty_train_table.data_editor(
                concat([session_state.x_train, session_state.y_train], axis=1),
                hide_index=False, disabled=True, width="stretch"
            )
            empty_test_title.markdown(f"Testing Set {len(session_state.x_test)} samples")
            empty_test_table.data_editor(
                concat([session_state.x_test, session_state.y_test], axis=1),
                hide_index=False, disabled=True, width="stretch"
            )

            if button("Clear the Dataset", type="secondary", width="stretch"):
                for key in page_sessions:
                    session_state[key] = None
                rerun()
