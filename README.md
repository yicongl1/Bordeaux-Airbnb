# 波尔多 Airbnb 房源价格预测 | Bordeaux Airbnb Price Prediction

**EDA · Visualization · Regression · Machine Learning**

> 课程 / Course: **BA08 « Data Science for the Smart City »** — Université de Technologie de Compiègne (UTC), Printemps 2024
> 小组 / Team: **Yicong LI · Yang LI**
> 交付形式 / Deliverable: Jupyter notebook (the written report *is* the notebook)

本项目用真实开放数据做一次完整的机器学习案例研究：从 Inside Airbnb 拿到波尔多（Bordeaux）的房源数据，做探索性分析（EDA）、可视化、统计检验，并比较四种回归模型对房源价格的预测能力。

This project is an end-to-end machine learning case study on real open data: we took the Bordeaux listings from Inside Airbnb, performed exploratory data analysis (EDA), visualization and statistical testing, and compared four regression models for nightly price prediction.

---

## 目录 / Table of Contents

- [1. 项目概览 / Overview](#1-项目概览--overview)
- [2. 数据集 / Dataset](#2-数据集--dataset)
- [3. 方法 / Methodology](#3-方法--methodology)
- [4. 结果 / Results](#4-结果--results)
- [5. 关键发现 / Key Findings](#5-关键发现--key-findings)
- [6. 仓库结构 / Repository Layout](#6-仓库结构--repository-layout)
- [7. 如何运行 / Getting Started](#7-如何运行--getting-started)
- [8. 局限与后续改进 / Limitations & Future Work](#8-局限与后续改进--limitations--future-work)
- [9. 参考资料 / References](#9-参考资料--references)
- [10. 致谢与联系 / Acknowledgements & Contact](#10-致谢与联系--acknowledgements--contact)

---

## 1. 项目概览 / Overview

### 中文

课程作业允许两种形式：从零构建一个智慧城市用例（Level 1），或复现并改进一个已有的公开案例（Level 2）。我们选择了 **Level 2**：以 Kaggle 上的 *NYC Airbnb: EDA, Visualization, Regression* 为基线，在**复现**其流程之后，提出并实现若干**改进**，再与基线对比评分。

与基线的主要差异：

1. **换城市、换数据源**：不使用 Kaggle 上二次加工的 NYC 数据，而是回到原始数据提供方 [Inside Airbnb](https://insideairbnb.com/get-the-data/)，选用**波尔多**的房源数据（同样的字段结构，更贴近“欧洲中型城市”的场景）。
2. **从文本里挖特征**：`name` 列中隐含着卧室数、床位数、浴室数与评分，我们用正则表达式把它们抽取成 5 个新特征。
3. **按价格分段建模**：房源价格明显分为低价与高价两类（大致以 100 € 为界），因此把数据集拆成“低价段（< 100 €）”和“高价段（> 100 €）”分别训练，而不是强行拟合一个全局模型。
4. **两套预处理 + 四个模型**：对比“数值化 + 独热编码”（第 4 章，Model 1）与“分位数分箱 + 独热编码”（第 5 章，Model 2），并在两套预处理上分别训练多元线性回归、随机森林、Ridge 与 Lasso。

### English

The assignment offered two formats: build a smart-city use case from scratch (Level 1), or **replay and improve** an existing public case (Level 2). We chose **Level 2**, using the Kaggle notebook *NYC Airbnb: EDA, Visualization, Regression* as the baseline: we replayed its pipeline, then proposed, implemented and benchmarked several improvements against it.

What we changed relative to the baseline:

1. **New city, original data source.** Instead of the pre-processed NYC dataset on Kaggle, we went back to [Inside Airbnb](https://insideairbnb.com/get-the-data/) and used the **Bordeaux** listings — same schema, but a European mid-sized city.
2. **Features mined from text.** The `name` column encodes the number of bedrooms, beds, baths and the guest rating. We extracted them into five new columns with regular expressions.
3. **Price-segmented modelling.** Listings fall into two clearly separated price regimes (around the 100 € boundary), so the data is split into a *low-price* (< 100 €) and a *high-price* (> 100 €) subset, each modelled separately.
4. **Two preprocessing pipelines × four models.** Numeric + one-hot encoding (chapter 4, "Model 1") versus quantile binning + one-hot encoding (chapter 5, "Model 2"), each with Multiple Linear Regression, Random Forest, Ridge and Lasso.

---

## 2. 数据集 / Dataset

### 中文

- 来源 / Source: [Inside Airbnb — Get the Data](https://insideairbnb.com/get-the-data/)（Bordeaux 快照）
- 文件 / File: `data/listings.csv`（约 2.2 MB，**latin-1** 编码；notebook 中依次尝试 `utf-8 → latin1 → gbk`）
- 规模 / Size: **11,854** 行 × **18** 列
- 目标变量 / Target: `price`（€ / 每晚），原始数据中 8,162 条价格非空，中位数 **82 €**、均值 **115 €**、最大值 **5,300 €**

字段说明：

| 列名 / Column | 含义 / Meaning |
| --- | --- |
| `id` | 房源 ID（建模时删除） |
| `name` | 房源标题，含“New / 2 bedrooms / 1 bath”等文本信息 |
| `host_id`, `host_name` | 房东信息（`host_name` 因隐私与伦理原因删除） |
| `neighbourhood_group` | 大区（Bordeaux, Mérignac, Talence, Pessac, Bègles …） |
| `neighbourhood` | 街区（62 个不同取值） |
| `latitude`, `longitude` | 经纬度，用于地图可视化 |
| `room_type` | Entire home/apt / Private room / Shared room / Hotel room |
| `price` | 每晚价格（€），**回归目标** |
| `minimum_nights` | 最少入住天数 |
| `number_of_reviews`, `reviews_per_month`, `number_of_reviews_ltm` | 评价相关计数 |
| `last_review` | 最后评价日期（删除） |
| `calculated_host_listings_count` | 该房东名下房源数 |
| `availability_365` | 一年中可预订天数 |
| `license` | 许可证号（删除） |

数据分布速览：

- 房源高度集中在 **Bordeaux**（6,663 套），其余为 Mérignac (726)、Talence (583)、Pessac (579)、Bègles (482)；为简化分析，只保留这 5 个房源最多的大区（共 9,033 套，约占 76%）。
- 房型以 **Entire home/apt (9,054)** 与 **Private room (2,704)** 为主。
- 清洗后用于建模的样本：**低价段 2,630 套**、**高价段 1,629 套**（共 4,259 套）。

### English

- Source: [Inside Airbnb — Get the Data](https://insideairbnb.com/get-the-data/) (Bordeaux snapshot)
- File: `data/listings.csv` (~2.2 MB, **latin-1** encoded; the notebook tries `utf-8 → latin1 → gbk`)
- Size: **11,854** rows × **18** columns
- Target: `price` (€ per night); 8,162 non-missing prices, median **€82**, mean **€115**, max **€5,300**

| Column | Meaning |
| --- | --- |
| `id` | Listing id (dropped before modelling) |
| `name` | Listing title, encodes "New / 2 bedrooms / 1 bath" etc. |
| `host_id`, `host_name` | Host info (`host_name` dropped for privacy/ethics) |
| `neighbourhood_group` | District (Bordeaux, Mérignac, Talence, Pessac, Bègles …) |
| `neighbourhood` | Neighbourhood (62 distinct values) |
| `latitude`, `longitude` | Coordinates, used for maps |
| `room_type` | Entire home/apt / Private room / Shared room / Hotel room |
| `price` | Nightly price (€), **regression target** |
| `minimum_nights` | Minimum stay |
| `number_of_reviews`, `reviews_per_month`, `number_of_reviews_ltm` | Review counts |
| `last_review` | Date of last review (dropped) |
| `calculated_host_listings_count` | Listings owned by the same host |
| `availability_365` | Bookable days per year |
| `license` | Licence number (dropped) |

At a glance:

- Listings concentrate in **Bordeaux** (6,663), then Mérignac (726), Talence (583), Pessac (579) and Bègles (482). We keep only these five largest groups (9,033 listings, ≈76%).
- Room types are dominated by **Entire home/apt (9,054)** and **Private room (2,704)**.
- Samples used for modelling after cleaning: **2,630** low-price and **1,629** high-price listings (4,259 in total).

---

## 3. 方法 / Methodology

### 3.1 特征工程 / Feature Engineering

**中文** — `name` 形如 `Rental unit in Bordeaux ¡¤ ¡ïNew ¡¤ 2 bedrooms ¡¤ 1 bath`。我们用正则表达式抽取 5 个新列：

- `note`：评分（“New” 记为 −1）；
- `number_of_bedrooms` / `number_of_beds` / `number_of_baths`：数量（`Studio` 记 1 间卧室，`Half-bath` 记 0.5 间浴室）；
- `type_of_baths`：`private` / `shared`。

**English** — `name` looks like `"Rental unit in Bordeaux ¡¤ ¡ïNew ¡¤ 2 bedrooms ¡¤ 1 bath"`. Five new columns are extracted with regex:

- `note`: guest rating (`"New"` → −1);
- `number_of_bedrooms` / `number_of_beds` / `number_of_baths` (`Studio` → 1 bedroom, `Half-bath` → 0.5 bath);
- `type_of_baths`: `private` / `shared`.

### 3.2 数据清洗 / Cleaning

**中文**：

- 删除 `id`、`host_name`、`last_review`、`license`、`name`。删除 `host_name` 不只是因为它无关紧要，也是出于**伦理考虑**：不应基于真实姓名对个人做推断，姓名对价格预测也没有信息量。
- `reviews_per_month` 缺失补 0；`price`、`note` 和三个房间容量特征缺失的行直接删除。
- 价格与计数类特征严重右偏，建模时取 **log10**（目标变量 `price` 同样取 log10）。

**English**:

- Drop `id`, `host_name`, `last_review`, `license`, `name`. Dropping `host_name` is not only about signal — it is an **ethical** choice: no inference should be made about real individuals, and names carry no predictive value for price.
- Missing `reviews_per_month` → 0; rows missing `price`, `note` or the three capacity features are dropped.
- Prices and count features are heavily right-skewed, so they are **log10**-transformed — including the target.

### 3.3 探索性分析与可视化 / EDA & Visualization

**中文** — 直方图 / 箱线图看分布与离群点；Folium 地图按大区与房型着色；相关系数热力图；按大区、房型、评论数、最少入住天数、可订天数等维度对比价格（并对 < 100 € 与 > 100 € 两个区间分别作图）；用**单因素方差分析（one-way ANOVA）+ 两两比较**检验“大区是否显著影响价格”。

**English** — Histograms/boxplots for distributions and outliers; Folium maps coloured by district and room type; correlation heatmap; price versus district, room type, reviews, minimum nights, availability, etc. (plotted separately for the < 100 € and > 100 € ranges); a **one-way ANOVA with pairwise comparisons** tests whether the district significantly affects price.

### 3.4 建模 / Modelling

**中文** — 两套预处理，分别在两个价格段上训练四个模型：

| 项目 / Item | 第 4 章 Model 1 | 第 5 章 Model 2 |
| --- | --- | --- |
| 数值特征 | log10 变换 | 按分位数 `qcut` 分箱（2–3 档） |
| 类别特征 | one-hot（`drop_first=True`） | one-hot（`drop_first=True`） |
| 特征数 | 88 | 57 |
| 模型 | 多元线性回归、随机森林 (`max_depth=8`, `n_estimators=100`)、Ridge、Lasso | 同左 |
| 调参 | — / GridSearchCV (`alpha`) | 同左 |
| 验证 | `KFold(n_splits=8/10, shuffle=True)`、70/30 划分 | 同左 |
| 指标 | RMSE、$R^2$（train / test） | 同左 |

**English** — Two preprocessing pipelines, each trained on both price segments with four models:

| Item | Chapter 4, Model 1 | Chapter 5, Model 2 |
| --- | --- | --- |
| Numeric features | log10 transform | quantile `qcut` bins (2–3 levels) |
| Categorical features | one-hot (`drop_first=True`) | one-hot (`drop_first=True`) |
| Feature count | 88 | 57 |
| Models | Multiple Linear Regression, Random Forest (`max_depth=8`, `n_estimators=100`), Ridge, Lasso | same |
| Tuning | — / `GridSearchCV` over `alpha` | same |
| Validation | `KFold(n_splits=8/10, shuffle=True)`, 70/30 split | same |
| Metrics | RMSE, $R^2$ (train / test) | same |

> 说明 / Note: 指标均在 **log10(price)** 空间上计算。RMSE = 0.09 表示典型相对误差约为 $10^{0.09}-1 \approx 23\%$。Metrics are computed on **log10(price)**; RMSE = 0.09 means a typical relative error of about $10^{0.09}-1 \approx 23\%$.

---

## 4. 结果 / Results

**中文** — 下表来自 `reports/model-performance.xlsx`（“4.1 与 5.1 数据处理对比”）。四种模型 × 两个价格段 × 两套预处理：

**English** — Below is the summary exported from `reports/model-performance.xlsx`. Four models × two price segments × two pipelines:

| Model | Segment | Preprocessing | RMSE (log10) | $R^2$ train | $R^2$ test |
| --- | --- | --- | --- | --- | --- |
| Linear | low (< 100 €) | 4.1 | 0.10 | 0.39 | **0.41** |
| Linear | low (< 100 €) | 5.1 | 0.11 | 0.34 | 0.25 |
| Linear | high (> 100 €) | 4.1 | 0.16 | 0.43 | **0.46** |
| Linear | high (> 100 €) | 5.1 | 0.17 | 0.32 | 0.28 |
| Random Forest | low (< 100 €) | 4.1 | **0.09** | 0.68 | **0.52** |
| Random Forest | low (< 100 €) | 5.1 | 0.10 | 0.59 | 0.41 |
| Random Forest | high (> 100 €) | 4.1 | **0.12** | 0.64 | **0.55** |
| Random Forest | high (> 100 €) | 5.1 | 0.13 | 0.63 | 0.43 |
| Ridge | low (< 100 €) | 4.1 | 0.10 | 0.41 | 0.33 |
| Ridge | low (< 100 €) | 5.1 | 0.11 | 0.33 | 0.26 |
| Ridge | high (> 100 €) | 4.1 | 0.15 | 0.43 | 0.39 |
| Ridge | high (> 100 €) | 5.1 | 0.17 | 0.32 | 0.29 |
| Lasso | low (< 100 €) | 4.1 | 0.15 | 0.43 | 0.39 |
| Lasso | low (< 100 €) | 5.1 | 0.17 | 0.32 | 0.29 |
| Lasso | high (> 100 €) | 4.1 | 0.15 | 0.43 | 0.39 |
| Lasso | high (> 100 €) | 5.1 | 0.17 | 0.32 | 0.29 |

*注：原表中部分 Lasso 行与 Ridge 行数值相同，此处如实转录。 / Some Lasso rows are numerically identical to Ridge rows in the source sheet; transcribed as-is.*

**结论 / Conclusion**：随机森林 + 第一套预处理（4.1）表现最好，低价段 $R^2_{test}=0.52$、高价段 $R^2_{test}=0.55$；线性类模型的 $R^2$ 都在 0.4 左右，说明价格与特征之间的关系存在明显的非线性，同时也受未观测因素（装修、景观、季节性、事件等）影响。第二套“分箱”预处理在所有模型上都**变差**，说明分箱丢失了信息。

Random Forest with pipeline 4.1 wins on both segments ($R^2_{test}=0.52$ and $0.55$). Linear models plateau around 0.4, which points to non-linear relationships plus unobserved drivers (decoration, view, seasonality, events). The binned pipeline (5.1) is **consistently worse** for every model — binning throws information away.

---

## 5. 关键发现 / Key Findings

**中文**

1. 房源价格差异极大（中位数 82 €，最高 5,300 €），且明显分为低价与高价两类；**按 100 € 分段建模**明显优于单一全局模型。
2. 与价格最相关的变量是 **大区（`neighbourhood_group`）**、**卧室数**、**床位数**、**浴室数**：
   - 大区的影响在 < 100 € 的房源上通过单因素方差分析与两两比较验证，p 值远小于 0.05；
   - 相关系数矩阵中，卧室数 / 床位数 / 浴室数与价格正相关。
3. 从 `name` 中抽取的三个容量特征是**关键特征**：去掉它们后，部分模型的 $R^2$ 甚至低于 0.1。
4. 随机森林在训练集与测试集上的 $R^2$ 差距较大（0.68 vs 0.52），存在过拟合，需要进一步调参 / 正则化。
5. 把连续变量分箱成类别（第 5 章）并没有提升效果，反而在各模型上都降低了 $R^2$。

**English**

1. Prices are extremely dispersed (median €82, max €5,300) and fall into two clearly separated regimes; **segmenting at 100 €** clearly beats a single global model.
2. The strongest price drivers are **district (`neighbourhood_group`)**, **bedrooms**, **beds** and **baths**:
   - the district effect on listings under 100 € is confirmed by one-way ANOVA and pairwise comparisons (p ≪ 0.05);
   - bedrooms, beds and baths correlate positively with price in the correlation matrix.
3. The three capacity features mined from `name` are **essential**: without them, some models fall below $R^2 = 0.1$.
4. Random Forest overfits (train 0.68 vs test 0.52 on the low-price segment) and needs further tuning/regularization.
5. Binning continuous variables into categories (chapter 5) does **not** help — every model loses $R^2$.

---

## 6. 仓库结构 / Repository Layout

```
Bordeaux-Airbnb/
├── README.md                     # 本文件（中英双语）/ this file (bilingual)
├── data/
│   └── listings.csv              # Inside Airbnb Bordeaux 原始数据 / raw listings (11,854 × 18)
├── notebooks/
│   └── ba08.ipynb                # 主交付物：完整报告 notebook / main deliverable (the report)
├── reports/
│   ├── Bordeaux Airbnb EDA, Visualization, Regression.html   # notebook 的 HTML 导出 / exported notebook
│   └── model-performance.xlsx    # 模型评分汇总 / model score summary
├── presentation/
│   ├── BA08-slides.pdf           # 答辩幻灯片（30 页）/ defence slides (30 pages)
│   └── BA08-projects-definition-P24.pdf   # 课程项目说明 / assignment brief
├── certificates/
│   ├── README.md                 # 证书清单、技能关键词与 LinkedIn 文案 / certificate list & LinkedIn blurbs
│   ├── Certificate(*).pdf        # 7 张 DataCamp 课程证书 / seven DataCamp certificates
│   └── DataCamp selection.pdf    # 课程推荐清单（Gilles MOREL, UTC）/ course reading list
```

**中文** — 仓库只包含本项目自己的代码与文档，不含任何第三方作品。证书的详细清单、技能关键词和可直接粘贴到 LinkedIn 的文案见 [`certificates/README.md`](certificates/README.md)。

**English** — The repository contains only our own code and documentation. See [`certificates/README.md`](certificates/README.md) for the full certificate list, skill keywords and copy-paste-ready LinkedIn entries.

---

## 7. 如何运行 / Getting Started

### 方式一：Google Colab（与原开发环境一致）/ Option 1: Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yicongl1/Bordeaux-Airbnb/blob/main/notebooks/ba08.ipynb)

**中文** — notebook 的第 3 个代码单元会挂载 Google Drive 并从固定路径读取数据：

```python
data = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/BA08/listings.csv', encoding=encoding)
```

因此需要把 `data/listings.csv` 上传到该路径，或修改这一行指向你自己的文件。

**English** — The third code cell mounts Google Drive and reads from a hard-coded path:

```python
data = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/BA08/listings.csv', encoding=encoding)
```

Upload `data/listings.csv` there, or point that line at your own copy.

### 方式二：本地 Jupyter / Option 2: Local Jupyter

```bash
pip install jupyter pandas numpy matplotlib seaborn scipy statsmodels scikit-learn folium
jupyter notebook notebooks/ba08.ipynb
```

**中文** — 本地运行时，请把第 3 个代码单元（`from google.colab import drive ...`）替换为：

```python
import pandas as pd
data = pd.read_csv('data/listings.csv', encoding='latin1')
```

其余单元（EDA、统计检验、建模）可原样运行。地图单元使用 `folium`，在 Colab 中可以内联显示，本地则会把地图写入 HTML。

**English** — When running locally, replace the third code cell (`from google.colab import drive ...`) with:

```python
import pandas as pd
data = pd.read_csv('data/listings.csv', encoding='latin1')
```

Everything else (EDA, statistical tests, modelling) runs unchanged. The map cells use `folium`; they render inline in Colab and write an HTML map locally.

**依赖 / Dependencies**: Python 3, `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`, `statsmodels`, `scikit-learn`, `folium` (and `google.colab` only for the Drive mount).

---

## 8. 局限与后续改进 / Limitations & Future Work

**中文**

- **样本量小**：清洗后仅 4,259 套房源，且高价段只有 1,629 套，模型方差较大。
- **数据快照**：只用了 Inside Airbnb 的单期快照，没有时间维度，无法建模季节性。
- **特征有限**：缺少面积、装修、景观、交通、评分明细等；`name` 的正则抽取对非标准标题会失效（会产生缺失并丢行）。
- **过拟合**：随机森林 train/test 差距明显，可尝试更深的超参搜索、交叉验证集成、XGBoost/LightGBM，或对目标做更稳健的变换。
- **可解释性**：可加入特征重要性 / SHAP 解释，回答“为什么这套房值这个价”。
- **评估口径**：$R^2$ 与 RMSE 在 log10 空间计算，与 Euro 空间的直觉不同，报告中应统一说明。

**English**

- **Small sample**: only 4,259 listings after cleaning (1,629 in the high-price segment), so model variance is high.
- **Single snapshot**: no time dimension, hence no seasonality modelling.
- **Limited features**: no surface area, renovation, view, transport or detailed ratings; the regex extraction from `name` fails on non-standard titles (creating NaNs and dropped rows).
- **Overfitting**: the Random Forest train/test gap is large; wider hyper-parameter search, ensembling, XGBoost/LightGBM or a more robust target transform are natural next steps.
- **Interpretability**: feature importance / SHAP would answer "why is this listing worth that price".
- **Metric scale**: $R^2$ and RMSE are computed in log10 space, which is not directly comparable to euro amounts — state this consistently in any report.

---

## 9. 参考资料 / References

- Inside Airbnb — open data: <https://insideairbnb.com/get-the-data/>
- Baseline case (Kaggle): <https://www.kaggle.com/code/wguesdon/nyc-airbnb-eda-visualization-regression>
- <https://www.kaggle.com/dgomonov/data-exploration-on-nyc-airbnb>
- <https://www.kaggle.com/biphili/hospitality-in-era-of-airbnb>
- <https://www.kaggle.com/chirag9073/airbnb-analysis-visualization-and-prediction>
- <https://www.kaggle.com/jrw2200/smart-pricing-with-xgb-rfr-interpretations>
- <https://www.kaggle.com/mpanfil/nyc-airbnb-data-science-ml-project>
- <https://www.kaggle.com/elcaiser/airbnb-data-exploration-on-nyc-visualization>
- <https://www.kaggle.com/nidaguler/eda>
- Course brief: `presentation/BA08-projects-definition-P24.pdf`

---

## 10. 致谢与联系 / Acknowledgements & Contact

**中文** — 感谢课程负责人 Gilles MOREL（UTC）的指导，也感谢 DataCamp 课程为数据分析与机器学习打下的基础（证书见 `certificates/`）。本项目可自由参考使用；如果用于学术或商业用途，请注明出处，欢迎与我们联系交流。

**English** — Thanks to Gilles MOREL (UTC) for the course guidance, and to the DataCamp tracks that built the foundations for this work (certificates in `certificates/`). Free to use and learn from; please cite this repository for academic or commercial reuse, and feel free to get in touch.

**Authors**: Yicong LI · Yang LI — June 2024
