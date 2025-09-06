#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/9/6 20:51
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   test.py
# @Desc     :   

from pandas import Series
from streamlit import (empty, sidebar, subheader, session_state, slider,
                       columns, metric, button, rerun, bar_chart,
                       markdown, plotly_chart)

from utils.helper import Timer, scatter_visualiser

empty_messages: empty = empty()
default_acc, default_pred, best_acc, best_pred = columns(4, gap="small")
default, best = columns(2, gap="small")
empty_charts_title: empty = empty()
empty_charts_chart: empty = empty()

home_sessions: list[str] = ["iris", "data", "X", "Y"]
for home_session in home_sessions:
    session_state.setdefault(home_session, None)
train_sessions: list[str] = ["x_train", "x_test", "y_train", "y_test", "model", "best"]
for train_session in train_sessions:
    session_state.setdefault(train_session, None)
test_sessions: list[str] = ["timer_test", "x_default", "y_default", "y_default_pred", "x_best", "y_best", "y_best_pred"]
for test_session in test_sessions:
    session_state.setdefault(test_session, None)

with sidebar:
    if session_state["data"] is None:
        empty_messages.error("Please upload a dataset in the Home page.")
    else:
        if session_state["model"] is None:
            empty_messages.info("Dataset uploaded. You should train a model in the train page.")
        else:

            subheader("Model Testing")

            index: int = slider(
                "Select Sample Index for Prediction",
                min_value=0, max_value=len(session_state.x_test) - 1, value=0, step=1,
                help="You can select the index of the sample from the test set for prediction.",
            )

            default_importance = Series(
                session_state.model.feature_importances_, index=session_state.x_train.columns
            ).sort_values(ascending=False)
            best_importance = Series(
                session_state.best.feature_importances_, index=session_state.x_train.columns
            ).sort_values(ascending=False)
            with default:
                markdown("**Feature Importance of Default Model**")
                bar_chart(default_importance, use_container_width=True)
            with best:
                markdown("**Feature Importance of Best Model**")
                bar_chart(best_importance, use_container_width=True)

            fig = scatter_visualiser(
                session_state.x_train,
                session_state.y_train,
                dims=2
            )
            empty_charts_title.markdown("**Training Set Visualisation (PCA 2D)**")
            empty_charts_chart.plotly_chart(fig, use_container_width=True)

            if session_state["y_default_pred"] is None:
                empty_messages.success("The model is ready for testing.")
                if button("Predict the Selected Sample", type="primary", width="stretch"):
                    with Timer("Predicting the selected sample with normal and best model") as t:
                        session_state["x_default"] = session_state.x_test.iloc[[index]]
                        session_state["y_default"] = session_state.y_test.iloc[index]
                        session_state["y_default_pred"] = session_state.model.predict(session_state.x_default)[0]

                        session_state["x_best"] = session_state.x_test.iloc[[index]]
                        session_state["y_best"] = session_state.y_test.iloc[index]
                        session_state["y_best_pred"] = session_state.best.predict(session_state.x_best)[0]
                    session_state["timer_test"] = t
                    rerun()
            else:
                empty_messages.success(f"{session_state.timer_test} Model testing completed.")

                with default_acc:
                    metric(
                        "Default Model: True Label",
                        f"{session_state["iris"].target_names[session_state.y_default][0]}",
                        delta=None, delta_color="normal",
                        help="The true label of the selected sample."
                    )
                with default_pred:
                    metric(
                        "Default Model: Predicted Label",
                        f"{session_state["iris"].target_names[session_state.y_default_pred]}",
                        delta=None, delta_color="normal",
                        help="The predicted label of the selected sample."
                    )

                with best_acc:
                    metric(
                        "Best Model: True Label",
                        f"{session_state["iris"].target_names[session_state.y_best][0]}",
                        delta=None, delta_color="normal",
                        help="The true label of the selected sample."
                    )
                with best_pred:
                    metric(
                        "Best Model: Predicted Label",
                        f"{session_state["iris"].target_names[session_state.y_best_pred]}",
                        delta=None, delta_color="normal",
                        help="The predicted label of the selected sample."
                    )

                if button("Clear the Dataset", type="secondary", width="stretch"):
                    for key in test_sessions:
                        session_state[key] = None
                    rerun()
