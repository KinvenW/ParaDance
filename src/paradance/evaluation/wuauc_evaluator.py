from typing import TYPE_CHECKING, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

if TYPE_CHECKING:
    from .calculator import Calculator

from .base_evaluator import evaluation_preprocessor


@evaluation_preprocessor
def calculate_wuauc(
    calculator: "Calculator",
    target_column: str,
    mask_column: Optional[str] = None,
    groupby: Optional[str] = None,
    weights_for_equation: List = [],
    weights_for_groups: Optional[pd.Series] = None,
    auc: bool = False,
) -> float:
    """Calculate weighted user AUC.

    :param groupby: groupby column
    :param weights_for_equation: weights for equation
    :param weights_for_groups: weights for group
    :param target_column: label column
    :param auc: bool, optional, default: False
    :return: AUC/WUAUC/UAUC
    """
    def safe_auc(y_true, y_score):
        try:
            return float(roc_auc_score(y_true, y_score))
        except ValueError:
            # 当只有一个类别时,返回0.5(随机猜测的AUC)
            return 0.5
    df = calculator.evaluated_dataframe
    if auc:
        result = float(safe_auc(df[target_column].values, df["overall_score"]))
    else:
        if groupby is not None:
            grouped = df.groupby(groupby).apply(
                lambda x: float(safe_auc(x[target_column], x["overall_score"]))
            )
            if weights_for_groups is not None:
                counts_sorted = weights_for_groups.loc[grouped.index]
                result = float(np.average(grouped, weights=counts_sorted.values))
            else:
                result = float(np.mean(grouped))
    return result
