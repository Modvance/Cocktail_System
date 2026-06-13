# cocktail_system

基于《认知与计算》课程第二题的数据集构建与目标说话人提取项目。

## 当前进度
当前已完成：
- 数据集构建目录骨架
- 源数据下载/解压脚本
- 源数据校验与报告脚本
- source manifest 生成脚本
- 生成数据、质检、preview 的初始骨架脚本

## 依赖
本项目统一在 conda 环境中进行，包括：
- 前面的数据集构建
- 当前模型训练 / 推理 / 评估
- 后续新增实验与脚本

当前环境文件：
- `environment.yml`：主 conda 环境配置

Python / 包依赖包括：
- Python 3.10
- `numpy`
- `PyYAML`
- `soundfile` / `pysoundfile`（真实音频读写必需；在 conda 环境中由 `pysoundfile` 提供）
- `Pillow`（preview 图像生成）
- `torch`（模型训练 / 推理必需）
- `torchaudio`（与 PyTorch 音频生态保持一致，便于后续扩展）
- 可选系统工具：`aria2c`、`wget`、`soxi`

创建并启用 conda 环境：

```bash
conda env create -f environment.yml
conda activate cocktail
```

如果后续你修改了环境文件，可以更新已有环境：

```bash
conda env update -f environment.yml --prune
conda activate cocktail
```

## 目录结构
- `environment.yml`：统一的 conda 环境配置
- `configs/sources.yaml`：原始数据源下载与目录配置
- `configs/dataset_build.yaml`：数据集构建参数配置
- `src/data/download_sources.py`：下载并解压原始语料
- `src/data/check_sources.py`：检查原始语料并生成报告
- `src/data/build_manifest.py`：扫描原始语料并生成 manifest CSV
- `src/data/build_tse_mix.py`：根据 manifest 生成 pilot TSE 音频样本
- `src/data/check_dataset.py`：检查已生成的数据目录、音频时长/峰值/活跃度、SNR/TIR 以及数据集级比例
- `src/data/make_preview.py`：固定随机抽样 preview 样本并生成波形图/频谱图
- `src/data/tse_dataset.py`：读取生成好的 TSE CSV 并构建训练 DataLoader
- `src/models/`：TSE-FAM 模型各子模块（特征提取、speaker encoder、mixture encoder、conditioning、mask、重建）
- `src/losses/`：SI-SDR 与频谱损失
- `src/debug/check_forward.py`：模型前向 / 反向 sanity check
- `src/debug/overfit_batch.py`：小批量过拟合检查
- `src/train.py`：训练与验证入口
- `src/infer.py`：单样本推理入口
- `src/evaluate.py`：测试集评估、分组统计、样例导出与结果图生成入口
- `src/metrics/metric_utils.py`：评估分组统计与 summary 聚合工具
- `src/visualize.py`：波形图、频谱图、mask、attention 与结果柱状图生成工具

## 使用方式
以下所有命令都默认在 `cocktail` conda 环境中执行。

### 1. 下载/解压数据源
```bash
python -m src.data.download_sources --config configs/sources.yaml
```

### 2. 检查数据源
```bash
python -m src.data.check_sources --config configs/sources.yaml
```

### 3. 生成 manifest
```bash
python -m src.data.build_manifest --config configs/sources.yaml
```

### 4. 生成 pilot 数据集
```bash
python -m src.data.build_tse_mix --config configs/dataset_build.yaml --dataset EN-TSE-Mix --splits train --num-samples 20
```

### 5. 检查生成结果
```bash
python -m src.data.check_dataset --dataset-root data/generated/EN-TSE-Mix
```

### 6. 生成 preview
```bash
python -m src.data.make_preview --dataset-root data/generated/EN-TSE-Mix --output-root results/dataset_preview
```

### 7. 模型前向检查
```bash
python src/debug/check_forward.py --config configs/train_en.yaml
```

### 8. 小批量过拟合检查
```bash
python src/debug/overfit_batch.py --config configs/train_en.yaml --num_samples 16 --steps 300
```

### 9. 训练 EN 模型
```bash
python src/train.py --config configs/train_en.yaml
```

### 10. 训练 ZH / 双语模型
```bash
python src/train.py --config configs/train_zh.yaml
python src/train.py --config configs/train_bilingual.yaml
```

### 11. 单样本推理
```bash
python src/infer.py \
  --ckpt checkpoints/en_tse_fam/best.pt \
  --mixture data/generated/EN-TSE-Mix/test/sample_000001/mixture.wav \
  --enrollment data/generated/EN-TSE-Mix/test/sample_000001/enrollment.wav \
  --out results/estimated_target.wav \
  --save_fig
```

### 12. 评估 EN 模型
```bash
python src/evaluate.py \
  --config configs/train_en.yaml \
  --ckpt checkpoints/en_tse_fam/best.pt \
  --test_csv data/generated/EN-TSE-Mix/test.csv \
  --out_dir results/eval/en_model_on_en
```

### 13. 使用评估配置文件
```bash
python src/evaluate.py --eval_config configs/eval_en.yaml
python src/evaluate.py --eval_config configs/eval_zh.yaml
python src/evaluate.py --eval_config configs/eval_cross_language.yaml
```

## 说明
当前 `build_tse_mix.py` 已具备完整的数据集构建逻辑：
- 基于 manifest 按 speaker 采样 target / enrollment / interferer
- 支持 2/3 人混合、`full_overlap` / `random_offset`、TIR 与 SNR 控制
- 支持 `EN-TSE-Mix`、`ZH-TSE-Mix` 与按样本单语采样的 `Bilingual-TSE-Mix`
- 会生成 `train.csv / valid.csv / test.csv / dataset_stats.json / quality_report.md`
- 会生成 `mixture.wav / target_clean.wav / enrollment.wav / noise.wav / meta.json`

当前说明：
- 依赖 `soundfile`，否则无法进行真实音频读写
- 数据构建、训练、推理、评估统一在 `cocktail` conda 环境中执行
- 训练 / 推理还依赖 `torch`
- bilingual 按方案采用“样本级中英混合”，即单条样本内部保持单语
- preview 会固定随机抽样，并生成 `waveform.png` 与 `spectrogram.png`
- 当前训练、推理、评估入口既支持 `python -m src...`，也支持 `python src/...` 直接运行
- `train.py` 当前已支持 AMP 开关、checkpoint 保存/恢复、`validate_every`/`train_metrics_every`/`save_examples_every` 降频控制，并默认只在 best checkpoint 更新时导出 `results/validation_samples/` 验证样例
- `infer.py` 支持 `--save_fig`，会额外输出 waveform、spectrogram、mask、attention 与 overview 图
- `evaluate.py` 会输出 `metrics_per_sample.csv`、`metrics_summary.json/.md`、按 speaker/SNR/lang/overlap 分组统计、`audio_examples/` 与 `figures/`
- 当前默认训练配置已针对服务器训练做了第一轮平衡：`batch_size=4`、`amp=true`、`num_workers=4`、`persistent_workers=true`、`validate_every=2`、只在 best epoch 导出验证样例
- 当前已补齐训练侧主干与可视化配套：Dataset、TSE-FAM、loss、debug、train、infer、evaluate、metric_utils、visualize
- 当前已在 `cocktail` conda 环境中完成动态验证：`check_forward.py` 通过、8 样本 overfit 检查通过、1 epoch smoke train 通过、推理 smoke test 通过、带配置文件的评估 smoke test 通过
- 当前 smoke 结果基于 `checkpoints/en_tse_fam_smoke/`、`results/estimated_target_smoke_v2.wav` 与 `results/eval/en_model_on_en_smoke_v2/`，说明最小训练 / 推理 / 评估 / 可视化链路已经跑通
