# LaTeX 实习报告排版说明

## 文件与编译

- 主文件：`main.tex`。以 pdfLaTeX 编译，完整命令为 `latexmk -pdf main.tex`；参考文献由 BibTeX 自动生成。
- 原始稿件：`prism-uploads/文字.docx`，未修改。
- 正文：`sections/` 中按顺序排列的七个 `.tex` 文件；第 7 节为 AI 辅助声明，位于结论之后、参考文献之前。摘要与封面在 `frontmatter/`。
- 图片：`figures/`，从 Word 中提取的 8 张 PNG 原图、后来补充的 2 张训练曲线和 2 张聚类数量选择曲线，以及索邦大学标志；未重绘或修改图中数据。
- 参考文献：`references.bib`，文末按首次引用顺序排列；正文使用自动编号引用。
- 附录：`appendices/implementation-details.tex` 和 `appendices/cluster-statistics.tex`。

## 排版处理

- 使用 A4 页面、11pt 正文字体、统一页眉页脚、目录和图表目录；封面不显示页码，前置部分使用罗马页码，正文使用阿拉伯页码。
- 保留英文稿件的研究内容、公式与数值。原稿从第 2 节跳到第 4 节，现按实际内容连续编为第 1–6 节；引言中的章节导读同步调整。
- 将原稿中的手写章节号、图表号和参考文献链接改为自动交叉引用，移除不可移植的 Windows 本地链接。
- 将 15 个独立公式转换为数学环境，按章节自动编号；6 张表格转换为可编辑的 LaTeX 表格。
- 正文保留输入通道表、方法对比表和 silhouette 结果表。
- 附录 A 收录网络逐层结构（表 A.1）与完整流程及训练参数（表 A.2）；附录 B 收录最终簇大小统计（表 B.1）。正文中保留相应讨论并链接到附录。
- 12 张研究图片组成 8 个编号图，其中图 4 含两个聚类数量选择子图，图 6 含三个训练子图，PCA 与 t-SNE 为同一图中的两个子图。

## 需要补充的信息

### 封面与页眉

封面仿照上传的 `prism-uploads/fe8af1efca9ce1ab7e47800f4cfd4d82.png`，仅保留索邦大学标志。使用项目内的 `prism-uploads/image.png` 标志原图，复制到 `figures/sorbonne-universite.png`；未使用样例中的巴黎西岱大学标志。

信息在 `frontmatter/metadata.tex` 中集中设置：

- 学校：Sorbonne Université。
- 项目：M2 – Parcours Statistique, apprentissage et algorithmes。
- 作者：Linshuo CHEN。
- 导师：Andrea CIARDI。

封面保留样例中的上下横线、居中标题、左侧作者和右侧导师布局；不显示未提供的日期、单位等占位信息。封面和页眉统一采用 Internship report，正文中的自称也改为 report。

### 训练图

图 6 已补齐，采用上方两图、下方居中一图的 a–c 布局：

- (a) `figures/ae-pretraining-loss.png`，对应上传的 `06_ae_pretraining_loss_spectrogram_k_auto_without_pz.png`。
- (b) `figures/dec-training-loss.png`，对应上传的 `07_dec_training_losses_spectrogram_k_auto_without_pz.png`。
- (c) `figures/dec-silhouette.png`，沿用 Word 中的 silhouette 曲线。

原 d 图（标签变化率）及其占位框已删除，图注和第 5.2 节的缺图说明也已替换。第 5.2 节补充了对现有曲线趋势的简短描述，没有根据图片编造精确数值。方法部分对标签变化率监测指标的定义仍保留；删除图位不等于删除该监测方法。

### 聚类数量选择图

第 3.4 节加入了两张并排的聚类数量选择曲线，合为图 4：`figures/cluster-count-elbow.png` 和 `figures/cluster-count-silhouette.png`。原始上传文件分别为 `06_ae_latent_k_selection_elbow_spectrogram_k_auto_without_pz.png` 和 `06_ae_latent_k_selection_silhouette_reference_spectrogram_k_auto_without_pz.png`。

正文和图注区分了肘部法选取的 K=5、仅供参考的预训练潜在空间 silhouette 曲线，以及 DEC 精化后的最终评分。后续图号、图表目录和交叉引用随编译自动更新。

### 参考文献

根据原稿已有链接核对了 Markidis 等人的轨迹分类论文、DEC、IDEC、silhouette 原始论文和 SciPy 官方文档的书目信息。未替换或扩写研究论证。

`moreiraReport` 对应的实习报告未随本次稿件提供，公开检索也未找到可确认的记录。目前保留原稿中的作者写法 `Medina Moreira` 和报告题目，并明确标记完整作者、单位与年份待确认；`n.d.` 表示日期未提供，不能当作已核实的书目信息。请提供该报告首页或完整引用后补齐。

SciPy 文档记录了查阅日期，但并未据此推断实验时的软件版本。若学校要求复现环境信息，仍需补充实际实验所用的版本。

### 内容核对范围

本次工作是稿件转换与排版，不是复现实验或全面的科学审稿。输入维度、网络层输出、超参数、评分和簇统计均按原稿保留；未利用图片反推或补造数据。