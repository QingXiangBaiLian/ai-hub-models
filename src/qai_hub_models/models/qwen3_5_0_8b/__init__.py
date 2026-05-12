# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.models._shared.llm.app import ChatApp as App  # noqa: F401
from qai_hub_models.models._shared.qwen3_5.model import (  # noqa: F401
    Qwen3_5PositionProcessor as PositionProcessor,
)

from .model import MODEL_ID  # noqa: F401
from .model import Qwen3_5_0_8B as FP_Model  # noqa: F401
from .model import Qwen3_5_0_8B_AIMETOnnx as Model  # noqa: F401
from .model import Qwen3_5_0_8B_QNN as QNN_Model  # noqa: F401
