---
name: POLCA-Net AI Assignment 3
overview: 以 AI 工具完成从算法检索到报告撰写的全流程：实现 2024 年论文 POLCA Net(基于自编码器的 PCA/LDA 非线性推广)并在 digits/MNIST/FashionMNIST 上与 PCA、LDA、普通 AE 对比,最后生成含 6 个指定章节的英文 .docx/.pdf 报告与 GitHub Pages 源码展示页。
todos:
  - id: skeleton
    content: 初始化仓库骨架、.gitignore、requirements.txt、种子与工具模块
    status: completed
  - id: core
    content: 实现 src/polca/losses.py(严格对齐论文五项损失)与 model.py
    status: completed
  - id: data-metrics
    content: 实现 data.py(digits/MNIST/FashionMNIST)与 metrics.py(NRMSE/PSNR/SSIM/正交性/方差排序)
    status: completed
  - id: train-eval
    content: 实现 baselines.py(PCA/LDA/AE)、train.py、evaluate.py
    status: completed
  - id: experiments
    content: 跑通全部实验与消融，产出 results/ 指标表与 figures/ 图表
    status: completed
  - id: prompts
    content: 撰写 AI_PROMPTS.md 记录全过程 AI 提示词
    status: completed
  - id: pages
    content: 生成 docs/index.html 源码展示页(GitHub Pages)
    status: completed
  - id: report
    content: 编写 report/build_report.py,生成英文 .docx 并经 Word COM 导出 .pdf
    status: completed
  - id: publish
    content: git init + 提交,输出 GitHub 推送与开启 Pages 的具体命令
    status: completed
  - id: finalize
    content: 回填 Pages 链接到报告第 6 节并重新导出 PDF
    status: completed
isProject: false
---

# POLCA-Net：AI 辅助完成 CISC3024 Assignment 3

## 0. 需求解析（来自 `requirement/`）

- `requirement/AIassign3.pdf`：目标是找到一个 **recent 的 PCA/LDA 算法**并应用于 CV/模式识别；要求 **完全由 AI 工具/agent 完成**算法检索、编程、报告写作；报告提交 UMMoodle，需过 Turnitin。
- `requirement/report_requirements.txt`：报告需含 6 部分 —— (1) 如何让 AI 找算法 (2) 算法描述 (3) AI 如何实现算法 (4) 实验设置与结果 (5) 从本次 AI 作业学到什么 (6) 源代码网页链接。
- 硬性约束：**不得自己手写代码**，所有代码由 AI 生成。

## 1. 算法选型（已通过检索确定）

**POLCA Net — Principal Orthogonal Latent Components Analysis Network**
arXiv:2410.07289 (2024-10-09), Martin H., Perozo F., Lopez M.

选它的理由：明确宣称"mimic and extend PCA and LDA capabilities to non-linear domains"，正好命中 "recent PCA/LDA algorithm + computer vision"；且有完整可复现的损失函数定义，便于 AI 精确实现与复现论文表格。

核心思想 = 自编码器 + 多目标复合损失：

```
L_polca = L_rec + L_class + alpha*L_ort + beta*L_com + gamma*L_var
```

| 分量 | 公式要点 | 对应 PCA/LDA 性质 |
|---|---|---|
| `L_rec` | 输入与重建的 MSE | 高保真重建/压缩 |
| `L_class` | 可选交叉熵（LDA 模式） | 类别可分性 |
| `L_ort` | `2/(n(n-1)) * sum_{i<j} S_ij^2`，`S = Z^T Z`，`z` 按 L2 归一化 | 隐空间正交性（PCA 核心） |
| `L_com` | `sum_i i^(1+eps) * E[(z_i - E z_i)^2] / (n * L_var)`，`eps=0.25` | 按方差排序潜维度 + 渐进式重建 |
| `L_var` | `sum_i E[(z_i - E z_i)^2]` | 防止 `L_com` 被"钻空子"，稳定训练 |

论文超参：`alpha=1e-2, beta=1e-2, gamma=1e-7`，batch=64，20k gradient updates，seed=5。
评价指标：NRMSE / PSNR / SSIM（重建）+ Accuracy / F1（Perceptron、Ridge、LogReg、Linear SVM 四个线性分类器）。

## 2. 仓库结构

```
AIassgin3/
├── src/polca/
│   ├── model.py        # POLCANet：encoder/decoder，线性/非线性 decoder 两种模式
│   ├── losses.py       # L_rec / L_ort / L_com / L_var / L_class（严格按论文公式）
│   ├── data.py         # digits(离线) / MNIST / FashionMNIST 加载与归一化
│   └── utils.py        # 随机种子(5)、设备选择、日志
├── src/baselines.py    # PCA、线性 LDA、普通 Autoencoder(无正交/排序损失)
├── src/metrics.py      # NRMSE、PSNR、SSIM、正交性(余弦相似度)、方差排序曲线
├── src/train.py        # POLCA 训练循环（含消融开关）
├── src/evaluate.py     # 4 个线性分类器 + Accuracy/F1；重建指标
├── src/make_figures.py # 全部图表输出
├── experiments/        # yaml/json 实验配置
├── results/            # csv/json 指标 + figures/ 图片
├── report/build_report.py      # python-docx 生成英文报告
├── report/AI_Assignment3_Report.docx + .pdf
├── docs/index.html     # GitHub Pages 源码展示页（项目页）
├── AI_PROMPTS.md       # 全流程 AI 提示词记录（报告第 1、3、5 节的原始素材）
├── README.md
└── requirements.txt
```

## 3. 实验设计

- **数据**：`load_digits`（sklearn 内置，100% 离线可跑）+ torchvision `MNIST` + `FashionMNIST`（已验证可下载，缓存到 `data/`）。图像统一 reshape 成 28x28 灰度、归一化到 [0,1]。
- **对比方法**：PCA(k=8/16/32)、LDA(supervised)、普通 Autoencoder、POLCA Net(unsupervised)、POLCA Net + 分类损失(supervised)。
- **重建指标**：NRMSE、PSNR、SSIM（`scikit-image`，需 `pip install scikit-image`；若失败则自实现 SSIM 兜底）。
- **分类指标**：4 个线性分类器 × Accuracy/F1。
- **PCA 性质验证**（报告亮点）：隐空间余弦相似度矩阵热力图（对比 PCA 应接近单位阵）、潜维度方差降序条形图（对比 PCA 解释方差曲线）。
- **消融**：逐项去掉 `L_ort` / `L_com` / `L_var`，量化各自贡献。
- **可复现**：全局 seed=5；结果以 CSV/JSON 落盘到 `results/`，脚本可一键重跑。

## 4. 图表产出（`results/figures/`）

`reconstruction_grid.png`（原图/PCA/AE/POLCA 对照）、`latent_orthogonality.png`、`variance_ordering.png`、`latent_scatter.png`（digits 二维嵌入）、`classification_metrics.png`（4 分类器对比柱状图）、`ablation.png`、`loss_curves.png`、`dataset_summary.png`。

## 5. 报告生成（英文，6 章节）

`report/build_report.py` 用 python-docx 组装：

1. **How AI Found the Algorithm** — 记录检索提示词与候选算法对比表（UMAP / t-SNE / Kernel PCA / σ-PCA / DFML / POLCA Net），为何选 POLCA Net。
2. **Algorithm Description** — PCA/LDA 回顾 + POLCA Net 架构图 + 五个损失项公式与超参。
3. **How AI Implements the Algorithm** — 目录结构、关键实现片段（loss 代码、模型类）、AI 协作流程（AI 生成 → 自动验证 → 修正）。
4. **Experiment Settings and Results** — 数据集、超参、硬件（RTX 4060 / PyTorch 2.14）、结果表格 + 全部图表 + 消融分析。
5. **What I Learned** — AI 工具在算法检索/编程/报告写作的作用、局限（幻觉、需人工验收）、可迁移方法论。
6. **Source Code Web Link** — GitHub Pages URL 占位 + 仓库地址。

排版：Times New Roman 12pt、1.5 倍行距、图表编号 `Fig. N`、表格编号 `Table N`、参考文献用 IEEE 风格引用 arXiv 2410.07289 及 PCA/LDA 原始文献。

**PDF 转换**：无 LibreOffice/pandoc，但本机装有 Microsoft Word 且 `pywin32` 可用 → 用 Word COM 自动化 (`win32com.client`) 把 .docx 导出为 PDF，保证与 docx 版式完全一致。

## 6. GitHub Pages 发布

本机无 `gh` CLI，仅有 `git`。执行方式：
1. 本地 `git init` + 提交全部代码（不含 `data/`、`results/figures` 大文件，加入 `.gitignore`）。
2. 生成 `docs/index.html`：项目介绍页 + 目录树 + 关键源码高亮 + 结果图表 + 到报告 PDF 的链接。
3. 需要用户配合：提供 GitHub 仓库名（或我先在本地建好并给出 `git remote add` 一行命令），推送后开启 Settings → Pages → Source = `main` / `docs`。
4. 最终链接形如 `https://<用户名>.github.io/<仓库名>/`，回填进报告第 6 节。

## 7. 执行顺序（todos）

1. 初始化仓库骨架、`.gitignore`、`requirements.txt`、seed 工具模块
2. 实现 `src/polca/losses.py`（严格对齐论文公式）与 `model.py`
3. 实现 `data.py`（digits / MNIST / FashionMNIST）与 `metrics.py`
4. 实现 `baselines.py`（PCA / LDA / AE）与 `train.py` / `evaluate.py`
5. 跑通全部实验，产出 `results/*.csv` 与 `results/figures/*.png`
6. 撰写 `AI_PROMPTS.md`（AI 提示词全过程记录）
7. 生成 `docs/index.html` 源码展示页
8. 编写 `report/build_report.py` → `.docx` → Word COM 转 `.pdf`
9. `git init` + 提交，输出 GitHub 推送与开 Pages 的具体命令
10. 回填 Pages 链接到报告第 6 节并重新导出 PDF

## 8. 风险与兜底

| 风险 | 兜底 |
|---|---|
| `scikit-image` 装不上 | 自实现 SSIM（高斯窗公式） |
| MNIST 下载失败 | 仅用 `digits` + 自造合成数据（正弦/弯曲图，论文本身也用） |
| Word COM 自动化失败（Word 未激活/宏限制） | 退回用 matplotlib PdfPages 或 `docx2pdf`，再不行直接交 `.docx`（PDF 为加分项） |
| GitHub 推送需凭证 | 提供本地完整仓库 + 一行 `git remote add`/`git push` 命令由用户执行 |
| GPU 训练波动 | seed 固定 + 结果取 3 次运行的均值/标准差 |

## 9. 交付物清单

- `report/AI_Assignment3_Report.docx` + `report/AI_Assignment3_Report.pdf`（英文，6 章节，含图表）
- `src/` 全部源码（AI 生成）+ `tests/smoke_test.py`
- `results/` 指标表与图、`docs/index.html` 源码展示页、`README.md`、`AI_PROMPTS.md`
- GitHub Pages 访问链接（写入报告第 6 节）
