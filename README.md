<p align="right">
  Language Switch / 语言选择：
  <a href="./README.zh-CN.md">🇨🇳 中文</a> | <a href="./README.md">🇬🇧 English</a>
</p>

**INTRODUCTION**
---
This application is designed to help users explore **supervised learning** using the classic **Iris dataset**. By
implementing a **decision tree classifier**, users can interactively visualize how the model **learns from the data**
and **makes predictions**. The app allows users to **select features**, **train the decision tree**, and understand key
concepts such as **entropy**, **Gini impurity**, **tree depth**, and **leaf nodes**. It is ideal for **beginners** who
want a **hands-on introduction** to **machine learning** and **decision tree algorithms**.

**DATA DESCRIPTION**
---
The Iris dataset is a classic multi-class classification dataset in machine learning. It contains **150 samples**,
divided into three classes of iris flowers (**Setosa, Versicolor, Virginica**). Each sample has **four features**: *
*sepal length**, **sepal width**, **petal length**, and **petal width**.

**HOW TO OBTAIN THE DATA**
---
You can directly load the Iris dataset using Python's `scikit-learn` library:

``` python
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data      # features
y = iris.target    # labels
feature_names = iris.feature_names
target_names = iris.target_names
```

**WEB DEVELOPMENT**
---

1. Install NiceGUI with the command `pip install streamlit`.
2. Run the command `pip show streamlit` or `pip show streamlit | grep Version` to check whether the package has been
   installed and its version.
3. Run the command `streamlit run app.py` to start the web application.

**PRIVACY NOTICE**
---
This application may require inputting personal information or private data to generate customised suggestions,
recommendations, and necessary results. However, please rest assured that the application does **NOT** collect, store,
or transmit your personal information. All processing occurs locally in the browser or runtime environment, and **NO**
data is sent to any external server or third-party service. The entire codebase is open and transparent — you are
welcome to review the code [here](./) at any time to verify how your data is handled.

**LICENCE**
---
This application is licensed under the [BSD-3-Clause License](LICENSE). You can click the link to read the licence.

**CHANGELOG**
---
This guide outlines the steps to automatically generate and maintain a project changelog using git-changelog.

1. Install the required dependencies with the command `pip install git-changelog`.
2. Run the command `pip show git-changelog` or `pip show git-changelog | grep Version` to check whether the changelog
   package has been installed and its version.
3. Prepare the configuration file of `pyproject.toml` at the root of the file.
4. The changelog style is [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/).
5. Run the command `git-changelog`, creating the `Changelog.md` file.
6. Add the file `Changelog.md` to version control with the command `git add Changelog.md` or using the UI interface.
7. Run the command `git-changelog --output CHANGELOG.md` committing the changes and updating the changelog.
8. Push the changes to the remote repository with the command `git push origin main` or using the UI interface.
