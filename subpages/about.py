#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/9/6 15:07
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   about.py
# @Desc     :   

from streamlit import title, expander, caption

title("**Application Information**")
with expander("About this application", expanded=True):
    caption("+ Evaluate model performance with accuracy and classification reports.")
    caption("+ Inspect feature importance for both default and optimized models.")
    caption("+ Visualize datasets in 2D or 3D PCA plots.")
    caption("+ Highlight individual samples and inspect predictions interactively.")
    caption("+ Simple and user-friendly Streamlit interface for all features.")
