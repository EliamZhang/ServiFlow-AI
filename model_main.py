"""模型推理入口：上游以 JSON dict 调用，返回 JSON 序列化结果 dict。

部署契约（对齐公司模型服务规范，见参考仓库 aus_old_risk_bid_submodel_v1_2_20260327_txn）：
- 推理对象路径：model_main.PredictMain
- __init__(model_dir)：模型加载必须在 __init__ 中完成，否则推理响应过慢甚至报错
- predict(input_dict) -> dict：入参格式与上游提前约定，出参必须是可 json 序列化的对象
- 由 uWSGI 托管

入参 key 列表：
    userId / applicationId / flowTime / bank_accounts / illion_raw_transactions
其中 illion_day_end_balances 等上游可能传入的附加 key 会被忽略。
出参结构与 verify_model.py 的输出完全同构（transactions + summaries + stats），
出错时返回 status="failed" 的结果 dict，不抛出异常。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from classification_core.service import ModelService


class PredictMain(object):
    """作用: 模型推理类"""

    def __init__(self, model_dir: str):
        """
        :param model_dir: string, 模型文件所在目录的绝对路径
        """
        # 模型（流水线配置 + 引擎规则）的加载必须在 __init__ 中完成，
        # 否则会导致推理响应过慢甚至报错。
        self._service = ModelService(
            pipeline_config_path=Path(model_dir) / "configs" / "pipeline.json",
            category_catalog_path=(
                Path(model_dir) / "configs" / "category_catalog.json"
            ),
        )

    def predict(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        """
        作用: 主模型推理方法
        :param input_dict: dict, 模型的入参，请与调用方提前沟通好入参格式，
            比如 key 的大小写，value 的类型等
        :return: 模型的出参，必须是可以 json 序列化的对象，如 list, dict 等
        """
        return self._service.predict(input_dict)
