from typing import List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pandas import DataFrame
from scipy import stats

from e1.constants import INDEX, RECENCY, OUTLIER, IQR, Z_SCORE_OUTLIER_THRESHOLD


def get_z_score_multivariate_outliers(data_frame: DataFrame) -> np.ndarray:
    data_frame = data_frame.iloc[:, 1:]
    z = np.abs(stats.zscore(data_frame))
    return np.where(z > Z_SCORE_OUTLIER_THRESHOLD)


def get_z_score_univariate_outliers(data_frame: DataFrame, variate: str) -> np.ndarray:
    z = np.abs(stats.zscore(data_frame[variate]))
    return np.where(z > Z_SCORE_OUTLIER_THRESHOLD)


def get_iqr_multivariate_outliers(data_frame: DataFrame) -> np.ndarray:
    data_frame = data_frame.iloc[:, 1:]
    q1 = data_frame.quantile(0.25)
    q3 = data_frame.quantile(0.75)
    iqr = q3 - q1
    outliers = np.where(
        ((data_frame < (q1 - 1.5 * iqr)) | (data_frame > (q3 + 1.5 * iqr))).any(axis=1)
    )
    return outliers


def get_iqr_univariate_outliers(data_frame: DataFrame, variate: str) -> np.ndarray:
    variate_data = data_frame[variate]
    q1 = variate_data.quantile(0.25)
    q3 = variate_data.quantile(0.75)
    iqr = q3 - q1
    return data_frame[
        ~((variate_data > (q1 - 1.5 * iqr)) | (variate_data < (q3 + 1.5 * iqr)))
    ]


class OutlierDetector:
    OUTLIER_DETECTION_METHODS = {
        IQR: get_iqr_multivariate_outliers,
    }

    @classmethod
    def remove_outliers(
        cls,
        data: DataFrame,
        outlier_detection_algorithm: str,
        column_names: List[str],
        plot_outliers_data: bool = True,
    ) -> DataFrame:
        outlier_detection_method = cls.OUTLIER_DETECTION_METHODS[
            outlier_detection_algorithm
        ]
        outliers_indexes = outlier_detection_method(data[column_names])
        data["outlier"] = np.where(np.isin(data[INDEX], outliers_indexes), "yes", "no")
        filtered_data = data.loc[data[OUTLIER] == "no"].drop(columns=OUTLIER)
        if plot_outliers_data is True:
            cls._plot_outliers_data(data, filtered_data, outlier_detection_algorithm)
        return filtered_data

    @classmethod
    def _plot_outliers_data(cls, data, filtered_data, outlier_detection_algorithm):
        plt.figure()
        plot_title = f"{outlier_detection_algorithm} outliers"
        sns.scatterplot(x=INDEX, y=RECENCY, data=data, hue=OUTLIER).set_title(
            plot_title
        )
        plt.figure()
        sns.relplot(x=INDEX, y=RECENCY, data=data, hue=OUTLIER, col=OUTLIER)
        plt.figure()
        sns.scatterplot(x=INDEX, y=RECENCY, data=filtered_data).set_title(
            "Data without outliers"
        )
