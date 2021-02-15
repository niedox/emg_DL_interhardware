from bioml import Align_Streams as AS
from bioml import Data_Cutter as DC
from bioml import Data_PreProcessing as DPP
from bioml import DB_connection as DBC
from bioml import Data_Loader as DL

AUGMENT = 0

class Raw_Data:
    def __init__(self, Tables, FSs, Labels, labels_name, recs):
        self.Tables = Tables
        self.FSs = FSs
        self.Labels = Labels
        self.labels_name = labels_name
        self.FSs = FSs
        self.recs = recs

class Split_Data:

    def __init__(self, Tables, FSs, Labels, labels_name, Train, Validate, Test, Train_Labels, Val_Labels, Test_Labels):
        self.Tables = Tables
        self.FSs = FSs
        self.Labels = Labels
        self.labels_name = labels_name

        self.Train = Train
        self.Validate = Validate
        self.Test = Test

        self.Train_Labels = Train_Labels
        self.Val_Labels = Val_Labels
        self.Test_Labels = Test_Labels

