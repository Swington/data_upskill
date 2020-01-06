import pandas as pd
from sklearn import preprocessing, linear_model, metrics

from e1.constants import (
    MONETARY,
    ATTRIBUTE_COLUMN_NAMES,
    LABEL,
    RANDOM_OVERSAMPLER,
    IQR,
)
from e1.data_resampler import DataResampler
from e1.data_scaler import DataScaler
from e1.outlier_detector import OutlierDetector

if __name__ == "__main__":
    test_data = pd.read_csv("test_data.csv")
    train_data = pd.read_csv("train_data.csv")

    train_data = train_data.drop(columns=MONETARY)

    train_data = OutlierDetector.remove_outliers(
        train_data, IQR, ATTRIBUTE_COLUMN_NAMES
    )

    train_data = DataScaler.scale(
        train_data, preprocessing.StandardScaler(), ATTRIBUTE_COLUMN_NAMES
    )

    attributes_data_resampled, label_data_resampled = DataResampler(
        RANDOM_OVERSAMPLER
    ).resample_data(train_data, ATTRIBUTE_COLUMN_NAMES, LABEL)

    linear_regression_model = linear_model.LinearRegression()

    linear_regression_model.fit(attributes_data_resampled, label_data_resampled)

    test_data = DataScaler.scale(
        test_data, preprocessing.StandardScaler(), ATTRIBUTE_COLUMN_NAMES
    )

    test_attributes_data = test_data[ATTRIBUTE_COLUMN_NAMES].values
    test_label_data = test_data[[LABEL]].values[:, 0]
    output_label_data = linear_regression_model.predict(test_attributes_data)

    fpr, tpr, thresholds = metrics.roc_curve(
        test_label_data, output_label_data, pos_label=1
    )
    auc = metrics.auc(fpr, tpr)
    print(auc)
