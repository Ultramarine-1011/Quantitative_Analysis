# 领域契约、金融语义与产物规范（Schema v0.2.0）

> 生效版本：`SCHEMA_VERSION = "0.2.0"`（兼容读取/显式迁移：`SUPPORTED_SCHEMA_VERSIONS = ("0.1.0", "0.2.0")`）
> 核心源码：[contracts.py](../contracts.py)、[artifacts/migration.py](../artifacts/migration.py)、[artifacts/store.py](../artifacts/store.py)

---

## 1. Schema 版本策略与 v0.1.0 -> v0.2.0 迁移规则

### 1.1 版本变更约定
- `SCHEMA_VERSION` 由 `"0.1.0"` 升级至 `"0.2.0"`。
- 任何改变对外 JSON 字段金融语义（例如 `max_drawdown` 符号约定）或产物寻址约定的改动，均通过版本升级 + 显式迁移完成，绝不静默更改同名字段含义。
- 缺失、非字符串、非标准 `MAJOR.MINOR.PATCH` 或不在 `SUPPORTED_SCHEMA_VERSIONS` 内的版本在加载时统一抛出 `UnsupportedSchemaVersionError`。

### 1.2 `max_drawdown` 与水下曲线语义统一
- **存储不变量（v0.2.0 起全命令一致）**：
  - `analyze`、`compare`、`portfolio`（含 `per_asset`、`benchmark.metrics`、`benchmark.performance`、`backtest`、`In_Sample_Portfolio`、`OOS_Portfolio`、`validation`、`sensitivity`）中的 `max_drawdown` 统一为**非负最大回撤深度（`>= 0.0`）**。
  - `underwater_drawdown` 时间序列保留非正水下表达式（`<= 0.0`）。
- **展现层规则（[reporting/formatting.py](../reporting/formatting.py)）**：
  - `format_drawdown_percent(value, *, decimals=2, na_text="N/A", input_convention="depth")` 默认要求非负深度（`>= 0`，允许 `-1e-12` 浮点误差），并在表格/报告中格式化为非正水下百分比（`<= 0%`）；遇到负值时抛出 `ValueError`，不再用 `-abs()` 静默掩盖符号冲突。
  - 若明确传入时间序列水下回撤，调用方必须显式指定 `input_convention="underwater"`（此时拒绝 `> 1e-12` 的正值）。

### 1.3 `0.1.0 -> 0.2.0` 显式迁移矩阵（`artifacts/migration.py`）
迁移函数 `migrate_result_payload(payload)` 严格按 `(schema_version, command)` 调度，**禁止盲目 `abs()`**：

| 来源版本 | 命令 (`command`) | Legacy 合法前提 | 迁移行为 (`0.1.0 -> 0.2.0`) | 违例行为 |
| --- | --- | --- | --- | --- |
| `0.1.0` | `compare` | `per_asset[*].max_drawdown <= 0` 且 `benchmark.metrics.max_drawdown <= 0` | 取反转换为非负深度 `>= 0`，写入 `methodology.max_drawdown_convention = "non_negative_depth"` | 若遇到 `> 1e-12` 正值，抛出 `SchemaMigrationError` |
| `0.1.0` | `analyze` | `metrics.max_drawdown >= 0` | 校验并归零 `[-1e-15, 0)` 微小噪声 | 若遇到 `< -1e-12` 负值，抛出 `SchemaMigrationError` |
| `0.1.0` | `portfolio` | `backtest`（除 `relative_to_equal_weight`）、`validation`、`sensitivity`、`benchmark.performance` 均 `>= 0` | 校验并写入 `methodology.max_drawdown_convention = "non_negative_depth"` | 若遇到 `< -1e-12` 负值，抛出 `SchemaMigrationError` |
| `0.1.0` | `fetch` | 无指标转换 | 提升 `schema_version` 为 `"0.2.0"` | 未知命令抛出 `ArtifactError` |
| `0.2.0` | 全部 | 按 v0.2.0 不变量校验 | 原样直通（`SchemaMigrationRecord.applied = False`） | 由 `validate_result_invariants` 阻断 |

---

## 2. 统一领域指标注册表（`METRIC_DEFINITIONS`）

[contracts.py](../contracts.py) 为单资产与多资产结果提供单一权威定义 `METRIC_DEFINITIONS`（共 37 项），并由 `build_metric_units` 与 `build_multi_asset_metric_units` 生成 `metric_units`；`validate_metric_value` 与 `validate_result_invariants` 在结果构造期自动执行。

| 指标键 | 单位 (`unit`) | 符号约定 (`sign_convention`) | 展现约定 (`display_convention`) | 年化基准 | 依赖收益口径 | `yield_level` 允许 |
| --- | --- | --- | --- | --- | --- | --- |
| `cumulative_return` / `total_return` | `ratio_decimal` | `signed` | `signed_percent` | `none` | False | False |
| `cagr` | `ratio_decimal_per_year` | `signed` | `signed_percent` | `calendar_365_25` | False | False |
| `max_drawdown` | `ratio_decimal` | `non_negative_depth` | `underwater_percent` | `none` | False | False |
| `underwater_drawdown` | `ratio_decimal` | `underwater_non_positive` | `underwater_percent` | `none` | False | False |
| `annualized_volatility` | `ratio_decimal_per_year` | `non_negative` | `percent` | `trading_days` | True | False |
| `sharpe_ratio` / `sortino_ratio` | `dimensionless_ratio` | `signed` | `ratio` | `trading_days` | True | False |
| `calmar_ratio` | `dimensionless_ratio` | `signed` | `ratio` | `calendar_365_25` | False | False |
| `var_95` / `cvar_95` | `ratio_decimal` | `signed` | `signed_percent` | `none` | False | False |
| `latest_close` | `price_units` | `signed` | `price_with_unit` | `none` | False | True |
| `latest_level` / `level_min` / `level_max` | `percentage_points_yield` | `signed` | `yield_percent` | `none` | False | True |
| `level_change` | `percentage_points_change` | `signed` | `yield_pp_signed` | `none` | False | True |
| `level_diff_std` | `percentage_points_change` | `non_negative` | `yield_pp` | `none` | False | True |
| `trading_days` / `periods_per_year` | `trading_days_per_year` | `positive` | `integer` | `none` | False | True |
| `return_convention` / `latest_signal` | `categorical` | `categorical` | `categorical` | `none` | False | True |
| `tracking_error` | `ratio_decimal_per_year` | `non_negative` | `percent` | `effective_periods` | True | False |
| `information_ratio` | `dimensionless_ratio` | `signed` | `ratio` | `effective_periods` | True | False |
| `beta` / `correlation` | `dimensionless_ratio` | `signed`（相关系数 `[-1, 1]`） | `ratio` | `none` | True | False |
| `r_squared` | `dimensionless_ratio` | `non_negative`（`[0, 1]`） | `ratio` | `none` | True | False |
| `jensen_alpha` | `ratio_decimal_per_year` | `signed` | `signed_percent` | `effective_periods` | True | False |
| `up_capture_ratio` / `down_capture_ratio` | `dimensionless_ratio` | `signed` | `percent` | `effective_periods` | True | False |
| `portfolio_volatility` | `ratio_decimal_per_year` | `non_negative` | `percent` | `effective_periods` | False | False |
| `expected_return` / `excess_annualized_return` | `ratio_decimal_per_year` | `signed` | `signed_percent` | `effective_periods` | False | False |
| `diversification_ratio` | `dimensionless_ratio` | `non_negative` | `ratio` | `none` | False | False |
| `effective_n_weights` / `effective_n_risk` | `effective_asset_count` | `positive` | `ratio` | `none` | False | False |
| `total_turnover` / `total_cost_ratio` | `ratio_decimal` | `non_negative` | `percent` | `none` | False | False |
| `excess_cumulative_return` | `ratio_decimal` | `signed` | `signed_percent` | `none` | False | False |

---

## 3. 结构化元数据契约与 `df.attrs` 一致性边界

1. **禁止隐式推断缺失复权**：对支持复权的股票/ETF 类别（`SUPPORTED_ADJUSTMENTS`：`ashare` / `etf` / `us_stock` / `hk_stock`），若参数与 `df.attrs` 同时缺失 `adjust` 或出现未支持复权值，`build_series_metadata` 立即抛出 `ValueError`。
2. **禁止冲突静默覆盖**：
   - `build_series_metadata` 拒绝参数与 `df.attrs` 间的 `asset_type`、非别名 `symbol`、`adjust`、`return_basis`、`series_kind` 与非兼容 `currency` 冲突。
   - `SeriesMetadata.__post_init__` 强校验 `adjust` vs `return_basis`、`trading_days >= 1`、`series_kind` 合法性以及 `yield_level`/`fx_rate` 不允许标记 `fx_converted=True`。
   - `AnalysisResult` / `MultiAssetResult` 在 `__post_init__` 中强制校验 `schema_version`、`command`、`status`、`asset_order` 一致性与 `validate_result_invariants`。
   - `yield_level` 序列在非 `error` 状态下禁止对 `YIELD_BLOCKED_METRICS`（9 项）输出非空数值，并必须附带 `metric_reasons[key] == "yield_level_not_return"`。
3. **双写一致性检验（`validate_frame_metadata_consistency` -> `sync_metadata_to_frame`）**：
   - 在 `services/analysis_service.py`、`services/preparation.py` 与 `artifacts/store.py` 构造或反序列化结果时，先执行 `validate_frame_metadata_consistency` 前置校验原始 `df.attrs` 与 `SeriesMetadata` 是否存在不一致（避免同步覆盖错误），再调用 `sync_metadata_to_frame` 将规范化 `SeriesMetadata` 与 `QualityReport` 双写至 `df.attrs`；此外先确定 `effective_adjust` 再推导 `return_basis`，修复历史覆盖缺陷。

---

## 4. 可移植产物包、Manifest 与路径安全（`artifacts/`）

1. **相对 POSIX 路径输出（v0.2.0）**：
   - `quant_cli.py::_write_outputs` 与 `artifacts.store.save_result_bundle` 输出的 `result.json["artifacts"]` 仅使用相对输出目录的 POSIX 路径：
     - `"output_dir": "."`
     - `"result_json": "result.json"`
     - `"data_parquet": "data.parquet"`、可选 `"data_csv": "data.csv"`、多资产 `"<frame>.parquet"`/`"<frame>.csv"`
     - 报告 `"report_md"` / `"report_html"`、图表 `"charts/<type>.<ext>"`
   - 仅当 `--export-csv` 未同时指定 `-o <dir>`（无落盘研究目录）时保留解析后路径；若与 `-o <dir>` 并用且位于输出目录外，不写入可移植 `result.json` 内部相对索引或 `artifact_manifest`。
2. **SHA-256 完整性清单（`artifact_manifest`）**：
   - `artifact_manifest` 结构为 `{"algorithm": "sha256", "files": {"<artifact_key>": {"path": "<rel_posix>", "sha256": "<hex>", "size_bytes": <int>}}}`，覆盖除 `output_dir` 与 `result_json` 外的全部研究目录内产出文件。
   - **v0.2.0 落盘产物包必需包含 `artifact_manifest`**：`save_result_bundle` 与 `quant_cli.py::_write_outputs` 始终生成该字段（即使 `files` 为空映射）；在默认 `verify_integrity=True` 加载 v0.2.0 产物时，缺失 `artifact_manifest` 或其值非映射对象一律抛出 `ArtifactIntegrityError`，绝不静默跳过 SHA-256 校验（仅调用方显式传入 `verify_integrity=False` 时跳过清单存在性与哈希校验）。
   - **v0.1.0 历史产物兼容**：v0.1.0 `result.json` 允许缺失 `artifact_manifest`（`None`）；若存在则会正常校验。
   - 加载时 `verify_artifact_manifest` 校验：`artifacts` 中每项（除 `output_dir` 与 `result_json`）均须存在于 `artifact_manifest.files`、路径字符串完全一致（防目录内路径替换）、文件存在且 SHA-256 与字节长度完全吻合；任一违例抛出 `ArtifactIntegrityError`。
3. **路径安全边界与声明读取策略（`resolve_safe_artifact_path` / `_load_declared_or_fallback_frame`）**：
   - 无条件拒绝 NUL 字节、`..` 路径段（同时拆解 `/` 与 `\`）、以及指向或穿越到 `output_dir` 之外的符号链接（包括 `result.json`、`data.parquet` 与 `charts/` 子目录）-> 抛出 `ArtifactSecurityError`。
   - **v0.2.0 严格禁止读取未声明文件**：`v0.2.0` 仅按 `result.json["artifacts"]` 显式声明的路径读取 Parquet/CSV，绝不回退读取研究目录内未声明的残留 `data.parquet` / `data.csv` / 多资产 frame 文件（无论 `verify_integrity` 为 `True` 或 `False`）；若声明的主 Parquet 文件缺失且未同时声明可用 CSV，立即抛出 `ArtifactIntegrityError`。
   - **v0.2.0** 严格禁止一切绝对路径与 `~` 前缀；仅 **v0.1.0** 允许受限兼容转换（**绝不访问外部原始绝对路径**，仅提取文件名或 `charts/<filename>` 映射到**当前指定 `output_dir` 内部**，支持 POSIX 与 Windows 盘符风格路径），并在 v0.1.0 `artifacts` 未显式声明 frame 路径时保留受 `resolve_safe_artifact_path` 保护的目录内默认文件安全回退。
   - `load_result_with_migration_info` 与 `load_result_from_artifacts`（[reporting/formatting.py](../reporting/formatting.py) 已薄委托至此）支持整个研究目录跨路径、跨设备复制并在原目录被篡改或删除后完全离线重放。

## M6-5 可选来源字段

- 已知成交量单位时，`SeriesMetadata.volume_unit` 记录供应商单位。历史结果可缺省本字段。`shares_or_units` 表示股或份，`lots_100` 表示每手 100 股或份；成交量原值保持原尺度。
- `adjustment_mismatch`、`volume_unit_mismatch`、`adjustment_unspecified` 标记来源口径冲突或缺失。C-REITs 的腾讯和东财直连路径标记前复权，新浪路径标记不复权。美元指数的跨币种折算以 `index_points_not_currency` 拒绝。期货异常 OHLC 计入逐资产质量报告，并影响研究状态。
- 启用输入快照时，`data_summary.snapshot_id` 为研究输入清单的 SHA-256。结果 Schema 仍为 0.2.0。读取历史产物无需补填来源信息。
