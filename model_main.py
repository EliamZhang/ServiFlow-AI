"""Model inference entry point: called by upstream with a JSON dict, returns a JSON-serializable result dict.

Deployment contract (aligned with the company model service convention; see the reference
repo aus_old_risk_bid_submodel_v1_2_20260327_txn):
- Inference object path: model_main.PredictMain
- __init__(model_dir): model loading must happen in __init__, otherwise inference
  responses become slow or even error out
- predict(input_dict) -> dict: input format is agreed with the caller in advance;
  the return value must be a JSON-serializable object
- Hosted by uWSGI

Input keys:
    userId / applicationId / flowTime / bank_accounts / illion_raw_transactions
Additional keys passed by upstream (e.g. illion_day_end_balances) are ignored.
Output structure is identical to verify_model.py's output (transactions + summaries + stats);
on error a status="failed" result dict is returned instead of raising an exception.
"""

from __future__ import annotations

from typing import Any

from classification_core.service import ModelService


class PredictMain(object):
    """Purpose: model inference class"""

    def __init__(self, model_dir: str):
        """
        :param model_dir: string, absolute path of the directory containing the model files
        """
        # Model loading (pipeline config + engine rules) must happen in __init__,
        # otherwise inference responses become slow or even error out.
        # This model loads no weights: configs and rule resources are resolved
        # relative to the code, so model_dir is intentionally ignored.
        self._service = ModelService()

    def predict(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        """
        Purpose: main model inference method
        :param input_dict: dict, model input; input format (e.g. key casing, value
            types) is agreed with the caller in advance
        :return: model output; must be a JSON-serializable object, e.g. list, dict
        """
        return self._service.predict(input_dict)
