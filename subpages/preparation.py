#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/9/6 15:07
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   preparation.py
# @Desc     :   

from pandas import DataFrame, concat, Series
from sklearn.datasets import load_iris
from streamlit import (empty, sidebar, subheader, session_state, button,
                       rerun, columns, metric)

from utils.helper import Timer

empty_messages: empty = empty()
total, left, mid, right = columns(4, gap="large")
empty_x_title: empty = empty()
empty_x_table: empty = empty()
empty_all_title: empty = empty()
empty_all_table: empty = empty()

all_sessions: list[str] = ["data", "timer_pre", "X", "Y"]
for all_session in all_sessions:
    session_state.setdefault(all_session, None)

with sidebar:
    if session_state.data is None:
        empty_messages.error("Please upload a dataset in the Home page.")
        subheader("Data Preparation")

        if button("Load the Dataset of Iris", type="primary", width="stretch"):
            with Timer("Loading the Iris dataset") as t:
                iris = load_iris()
                features = iris.data
                labels = iris.target

                session_state["X"]: DataFrame = DataFrame(features, columns=iris.feature_names)
                session_state["Y"]: DataFrame = DataFrame(labels, columns=["target"])
                session_state["data"] = concat([session_state.X, session_state.Y], axis=1)
            session_state["timer_pre"] = t
            rerun()
    else:
        empty_messages.success(f"{session_state.timer_pre} Dataset loaded successfully.")

        empty_x_title.markdown("Features Set as X")
        empty_x_table.data_editor(session_state.X, hide_index=False, disabled=True, width="stretch")
        empty_all_title.markdown("The Data with X and Y")
        empty_all_table.data_editor(session_state.data, hide_index=False, disabled=True, width="stretch")

        count: Series = session_state["Y"].value_counts()
        categories: list[int] = count.index.tolist()
        values: list[int] = count.values.tolist()
        with total:
            metric("Total Samples", sum(values))
        with left:
            metric(f"Category {categories[0]}", values[0])
        with mid:
            metric(f"Category {categories[1]}", values[1])
        with right:
            metric(f"Category {categories[2]}", values[2])

        if button("Clear the Dataset", type="secondary", width="stretch"):
            for key in all_sessions:
                session_state[key] = None
            rerun()
