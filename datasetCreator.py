import os, json
import pandas as pd
from sklearn.model_selection import train_test_split
from torch import tensor
from torch.utils.data import Dataset,RandomSampler, SequentialSampler
from torch.utils.data.dataloader import DataLoader





class DatasetCreator():
    """
    Class for loading data and doing traint-test split from dataset. It also creates loaders for evaluating and testing.

    Args:
        Dataset (string): [description]
    """

    def __init__(self,dataset,load=False, workers=0, batch_size=64):
        """

        Initialization loads the data into memory

        Args:
            dataset (string): path to the root of dataset
            load (bool, optional): If load indexs from file. Defaults to None.
        """
        class _InnerDataset(Dataset):
            """
            Helpfull inner class for creating datasets of splitted data.
            """
            def __init__(self,data):
                """
                initialize the data
                Args:
                    data (dict): dictionary which holds the data
                """
                self._data = data

            def __getitem__(self, index):
                return (tensor([self._data.iloc[index]["time"], self._data.iloc[index]["en_total"], self._data.iloc[index]["pe_total"], self._data.iloc[index]["be_total"],
                 self._data.iloc[index]["pressure"], self._data.iloc[index]["s_xx"], self._data.iloc[index]["s_xy"], self._data.iloc[index]["s_xz"],
                 self._data.iloc[index]["s_yx"], self._data.iloc[index]["s_yy"], self._data.iloc[index]["s_yz"], self._data.iloc[index]["s_zx"], self._data.iloc[index]["s_zy"], 
                 self._data.iloc[index]["s_zz"], self._data.iloc[index]["temperature"], self._data.iloc[index]["bndlen_av"], self._data.iloc[index]["bndlen_max"], self._data.iloc[index]["bndlen_min"]]),
                 self._data.iloc[index]["label"])


            def __len__(self):
                return len(self._data["time"])


        values ={"time": [], "en_total": [], "pe_total": [], "be_total": [], "pressure": [], "s_xx": [], "s_xy": [], "s_xz": [], "s_yx": [], "s_yy": [],
         "s_yz": [], "s_zx": [],"s_zy": [], "s_zz": [], "temperature": [], "bndlen_av": [], "bndlen_max": [], "bndlen_min": [], "label" :[]}
        self._batch_size = batch_size
        self._workers = workers
        for cur_dir in os.scandir(dataset):
            if not cur_dir.name.startswith('.') and cur_dir.is_dir():
                for dpd in os.scandir(os.path.join(dataset,cur_dir)):
                    time = 0.0
                    for sim in sorted(os.listdir(os.path.join(dataset,cur_dir,dpd))):
                        if sim != '50':
                            label=0
                            if os.path.isfile(os.path.join(dataset,cur_dir,dpd,sim,'1')):
                                label=1
                            try:
                                with open(os.path.join(dataset,cur_dir,dpd,sim,"CORREL")) as f:
                                    for row in f.readlines()[1:]:
                                        values["label"].append(label)
                                        dict_iter= iter(list(values.keys())[1:-1])
                                        first = True
                                        for val in iter(row.split(" ")):
                                                if val:
                                                    if first:
                                                        time += float(val)
                                                        values["time"].append(time)
                                                        first = False
                                                    else:
                                                        values[next(dict_iter)].append(float(val))
                            except:
                                None
        dataframe = pd.DataFrame(values)
        if load:
            with open("index_split.json", "r") as f:
                dict = json.load(f)
            dataframe_train = dict["train"]
            dataframe_test= dict["test"]
        else:
            dataframe_train, dataframe_test = train_test_split(list(range(len(dataframe))), train_size=0.8, test_size=0.2, stratify=dataframe["label"])
            with open("index_split.json", "w") as f:
                json.dump({"train": dataframe_train, "test":dataframe_test},f)
        self._dataframe_train = _InnerDataset(dataframe.iloc[dataframe_train])
        self._dataframe_test = _InnerDataset(dataframe.iloc[dataframe_test])
        return


    def loader_train(self):
        """
        Create loader which is suppose to be used for training

        Args:
            batch_size (int, optional): size of the batches. Defaults to 64.

        Returns:
            iter: iterator which will create batches of data.
        """
        return DataLoader(self._dataframe_train,batch_size= self._batch_size, sampler=RandomSampler(data_source=self._dataframe_train), num_workers=self._workers)


    def loader_evaluate_train(self):
        """
        Create loader which is supposed to be used for evaluation of train dataset.

        Args:
            batch_size (int, optional): size of the batches. Defaults to 64.

        Returns:
            iter: iterator which will create batches of data.
        """
        return DataLoader(self._dataframe_train,batch_size= self._batch_size, sampler=SequentialSampler(data_source=self._dataframe_train), num_workers=self._workers) 


    def loader_evaluate_test(self):
        """
        Create loader which is supposed to be used for evaluation of test dataset

        Args:
            batch_size (int, optional): size of the batches. Defaults to 64.

        Returns:
            iter: iterator which will create batches of data.
        """
        return DataLoader(self._dataframe_test, batch_size= self._batch_size, sampler=SequentialSampler(data_source=self._dataframe_test), num_workers=self._workers) 
