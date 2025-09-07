<p align="right">
  Language Switch / 语言选择：
  <a href="./README.zh-CN.md">🇨🇳 中文</a> | <a href="./README.md">🇬🇧 English</a>
</p>

**应用简介**
---
本应用旨在帮助用户探索**监督学习**，使用经典的**鸢尾花（Iris）数据集**。通过实现**决策树分类器**，用户可以交互式地可视化模型如何
**从数据中学习**并**进行预测**。该应用允许用户**选择特征**、**训练决策树**，并理解关键概念，如**熵（entropy）**、**基尼不纯度（Gini
impurity）**、**树的深度（tree depth）**和**叶节点（leaf nodes）**。本应用非常适合**初学者**进行**动手实践**，学习**机器学习**和
**决策树算法**。用户可以交互式选择训练/测试集、配置模型参数，并查看默认模型和优化模型的表现。

**数据简介**
---
鸢尾花数据集是机器学习中经典的多类别分类数据集，包含**150 条样本数据**，分为三类鸢尾花（Setosa、Versicolor、Virginica）。每条样本有
**四个特征**：**花萼长度（sepal length）**、**花萼宽度（sepal width）**、**花瓣长度（petal length）**、**花瓣宽度（petal width）**。

**功能特性**
---

- **数据加载与探索**：加载鸢尾花数据集，查看特征表格，并显示类别分布指标。
- **训练/测试集划分**：可自定义训练/测试集比例和随机种子。
- **决策树训练**：使用 Gini 或 Entropy 准则训练决策树分类器。
- **超参数优化**：通过 GridSearchCV 自动搜索最优决策树超参数。
- **模型评估**：在测试集上评估模型性能，包括准确率、分类报告和特征重要性。
- **交互式可视化**：在二维或三维 PCA 图中可视化训练/测试数据，可高亮选定样本。
- **Streamlit 用户界面**：简洁交互式网页界面，支持滑块、按钮和动态图表。

**如何获取数据**
---
可以通过 Python 的 `scikit-learn` 库直接下载鸢尾花数据集：

``` python
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data      # 特征
y = iris.target    # 标签
feature_names = iris.feature_names
target_names = iris.target_names
```

**网页开发**
---

1. 使用命令`pip install streamlit`安装`Streamlit`平台。
2. 执行`pip show streamlit`或者`pip show git-streamlit | grep Version`检查是否已正确安装该包及其版本。
3. 执行命令`streamlit run app.py`启动网页应用。

**隐私声明**
---
本应用可能需要您输入个人信息或隐私数据，以生成定制建议和结果。但请放心，应用程序 **不会**
收集、存储或传输您的任何个人信息。所有计算和数据处理均在本地浏览器或运行环境中完成，**不会** 向任何外部服务器或第三方服务发送数据。

整个代码库是开放透明的，您可以随时查看 [这里](./) 的代码，以验证您的数据处理方式。

**许可协议**
---
本应用基于 **BSD-3-Clause 许可证** 开源发布。您可以点击链接阅读完整协议内容：👉 [BSD-3-Clause License](./LICENSE)。

**更新日志**
---
本指南概述了如何使用 git-changelog 自动生成并维护项目的变更日志的步骤。

1. 使用命令`pip install git-changelog`安装所需依赖项。
2. 执行`pip show git-changelog`或者`pip show git-changelog | grep Version`检查是否已正确安装该包及其版本。
3. 在项目根目录下准备`pyproject.toml`配置文件。
4. 更新日志遵循 [Conventional Commits](https://www.conventionalcommits.org/zh-hans/v1.0.0/) 提交规范。
5. 执行命令`git-changelog`创建`Changelog.md`文件。
6. 使用`git add Changelog.md`或图形界面将该文件添加到版本控制中。
7. 执行`git-changelog --output CHANGELOG.md`提交变更并更新日志。
8. 使用`git push origin main`或 UI 工具将变更推送至远程仓库。
