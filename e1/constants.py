import datetime

TRANSFUSION_DATA_FILENAME = "transfusion.csv"
DATA_TIMESTAMP = datetime.date(2008, 10, 3)

# Column names
INDEX = "Index"
RECENCY = "Recency (months)"
FREQUENCY = "Frequency (times)"
MONETARY = "Monetary (c.c. blood)"
TIME = "Time (months)"
OUTLIER = "outlier"

# Outlier detection algorithms
IQR = "IQR"

# Arbitrary outlier detection values
Z_SCORE_OUTLIER_THRESHOLD = 3

# Training names
ATTRIBUTE_COLUMN_NAMES = [RECENCY, FREQUENCY, TIME]
LABEL = "whether he/she donated blood in March 2007"

# oversampler types
RANDOM_OVERSAMPLER = "random"
