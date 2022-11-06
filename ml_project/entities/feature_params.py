from dataclasses import dataclass
from typing import List, Optional


@dataclass
class FeatureParams:
    _target_: str = "ml_project.preprocessing.transformer.Transformer"
    categorical_features: Optional[List[str]] = None
    numeric_features: Optional[List[str]] = None
