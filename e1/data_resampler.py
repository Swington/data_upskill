from typing import Tuple, List

from imblearn.over_sampling import RandomOverSampler
from pandas import DataFrame

from e1.constants import RANDOM_OVERSAMPLER


class DataResampler:
    OVERSAMPLER_MAPPING = {
        RANDOM_OVERSAMPLER: RandomOverSampler(random_state=0),
    }

    def __init__(self, oversampler_type: str):
        self.oversampler = self.OVERSAMPLER_MAPPING[oversampler_type]

    def resample_data(
        self,
        data: DataFrame,
        attribute_column_names: List[str],
        label_column_name: List[str],
    ) -> Tuple[DataFrame, DataFrame]:
        attributes_values = data[attribute_column_names].values
        label_values = data[[label_column_name]].values[:, 0]
        attributes_data_resampled, label_data_resampled = self.oversampler.fit_resample(
            attributes_values, label_values
        )
        return attributes_data_resampled, label_data_resampled
