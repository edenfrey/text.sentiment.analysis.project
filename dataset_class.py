class Dataset:
    def __init__(self, DATA, DATASET_COLUMN_NAMES, DATASET_ENCODING: str = "ISO-8859-1", TRAIN_SIZE: float = 0.8 ) -> bool:
        self.DATA = DATA
        self.DATASET_COLUMN_NAMES = DATASET_COLUMN_NAMES
        self.DATASET_ENCODING = DATASET_ENCODING
        self.TRAIN_SIZE = TRAIN_SIZE   
        self.TEXT_CLEANING_FORMAT = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"