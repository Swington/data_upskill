from typing import List

import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame

from e1.constants import RECENCY


class DataScaler:
    @classmethod
    def scale(
        cls,
        data: DataFrame,
        scaler: object,
        column_names: List[str],
        plot_scaled_data: bool = True,
    ) -> DataFrame:
        data_to_scale = data[column_names]
        scaler = scaler.fit(data_to_scale)
        data[column_names] = scaler.transform(data_to_scale)
        if plot_scaled_data is True:
            plt.figure()
            sns.scatterplot(
                x=range(len(data_to_scale)), y=RECENCY, data=data_to_scale
            ).set_title("Scaled transfusion recency data")
        return data
