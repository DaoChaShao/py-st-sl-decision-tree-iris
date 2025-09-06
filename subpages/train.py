#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/9/6 15:52
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   train.py
# @Desc     :   

from graphviz import Source
from pandas import DataFrame, concat
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
from streamlit import (empty, sidebar, subheader, session_state, slider,
                       caption, number_input, button, rerun, selectbox,
                       columns, metric, data_editor, markdown, graphviz_chart,
                       text)

from utils.helper import Timer, tree_model_seeker

empty_messages: empty = empty()
normal, best = columns(2, gap="large")

empty_tree_title: empty = empty()
empty_tree_chart: empty = empty()
empty_tree_text: empty = empty()

empty_best_title: empty = empty()
empty_best_chart: empty = empty()
empty_best_text: empty = empty()

empty_train_title: empty = empty()
empty_train_table: empty = empty()
empty_test_title: empty = empty()
empty_test_table: empty = empty()

all_sessions: list[str] = ["iris", "data", "X", "Y"]
for all_session in all_sessions:
    session_state.setdefault(all_session, None)
page_sessions: list[str] = ["x_train", "x_test", "y_train", "y_test", "timer_split", "model", "timer_train", "best"]
for page_session in page_sessions:
    session_state.setdefault(page_session, None)

with (sidebar):
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
                    (
                        session_state["x_train"], session_state["x_test"],
                        session_state["y_train"], session_state["y_test"]
                    ) = train_test_split(
                        session_state.X, session_state.Y,
                        test_size=size / 100,
                        random_state=seed,
                        shuffle=True,
                        stratify=None
                    )
                session_state.timer_split = t
                rerun()
        else:
            empty_train_title.markdown(f"**Training Set {len(session_state.x_train)} samples**")
            empty_train_table.data_editor(
                concat([session_state.x_train, session_state.y_train], axis=1),
                hide_index=False, disabled=True, width="stretch"
            )
            empty_test_title.markdown(f"**Testing Set {len(session_state.x_test)} samples**")
            empty_test_table.data_editor(
                concat([session_state.x_test, session_state.y_test], axis=1),
                hide_index=False, disabled=True, width="stretch"
            )

            criteria: list[str] = ["gini", "entropy", ]
            criterion: str = selectbox(
                "Split Criterion",
                options=criteria,
                index=1,
                help="The function to measure the quality of a split."
            )
            caption(f"The split criterion you selected is **{criterion}**.")

            if session_state["model"] is None:
                empty_messages.info(
                    f"{session_state.timer_split} Dataset split successfully. You can train the model now."
                )

                if button("Train the Model", type="primary", width="stretch"):
                    with Timer("Training the Decision Tree Classifier") as t:
                        session_state["model"] = DecisionTreeClassifier(criterion=criterion, random_state=seed)
                        session_state["model"].fit(session_state.x_train, session_state.y_train)
                        session_state["best"], best_params, _ = tree_model_seeker(
                            session_state.x_train, session_state.y_train, randomness=seed
                        )
                        print(f"The best hyperparameters are: {best_params}")
                    session_state["timer_train"] = t
                    rerun()
            else:
                empty_messages.success(
                    f"{session_state.timer_train} Dataset split successfully. The model has been trained."
                )

                y_pred = session_state["model"].predict(session_state.x_test)
                report = classification_report(session_state.y_test, y_pred, output_dict=True)
                report = DataFrame(report).transpose()
                accuracy = accuracy_score(session_state.y_test, y_pred)
                percentage: float = round(accuracy * 100, 2)
                with normal:
                    metric(
                        "Normal Accuracy on Test Set",
                        f"{percentage} %", delta=f"{percentage - 100:.4f} %",
                        delta_color="normal"
                    )

                    data_editor(report, hide_index=False, disabled=True, width="stretch")

                    markdown("**Decision Tree Visualization**")
                    dot_data = export_graphviz(
                        session_state["model"],
                        out_file=None,
                        feature_names=session_state["iris"].feature_names,
                        class_names=session_state["iris"].target_names,
                        filled=True,
                        rounded=True,
                        special_characters=True
                    )
                    graph = Source(dot_data)
                    graphviz_chart(graph, width="stretch")

                    text(export_text(
                        session_state["model"],
                        feature_names=session_state["iris"].feature_names,
                        show_weights=False,
                    ))

                y_best = session_state["best"].predict(session_state.x_test)
                best_report = classification_report(session_state.y_test, y_best, output_dict=True)
                best_report = DataFrame(best_report).transpose()
                best_accuracy = accuracy_score(session_state.y_test, y_best)
                best_percentage: float = round(best_accuracy * 100, 2)
                with best:
                    metric(
                        "Best Accuracy on Test Set",
                        f"{best_percentage} %", delta=f"{best_percentage - 100:.4f} %",
                        delta_color="normal"
                    )

                    data_editor(best_report, hide_index=False, disabled=True, width="stretch")

                    markdown("**Best Decision Tree Visualization**")
                    best_dot_data = export_graphviz(
                        session_state["best"],
                        out_file=None,
                        feature_names=session_state["iris"].feature_names,
                        class_names=session_state["iris"].target_names,
                        filled=True,
                        rounded=True,
                        special_characters=True
                    )
                    best_graph = Source(best_dot_data)
                    graphviz_chart(best_graph, width="stretch")

                    text(export_text(
                        session_state["best"],
                        feature_names=session_state["iris"].feature_names,
                        show_weights=False,
                    ))

            if button("Clear the Dataset", type="secondary", width="stretch"):
                for key in page_sessions:
                    session_state[key] = None
                rerun()
