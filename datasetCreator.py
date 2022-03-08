from sklearn.preprocessing import StandardScaler

import os, json,torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset,Sampler, SequentialSampler, RandomSampler
from torch.utils.data.dataloader import DataLoader

class _InnerDataset(Dataset):
            """
            Helpfull inner class for creating datasets of splitted data.
            """
            def __init__(self,data, labels, device="cpu"):
                """
                initialize the data
                Args:
                    data (dict): dictionary which holds the data
                """
                
                self._columns = len(DatasetCreator.CHOSEN_COLUMNS)
                self._data = torch.tensor(np.swapaxes(data[DatasetCreator.CHOSEN_COLUMNS].values,0,1)).to(device)
                self._label = torch.tensor(labels).to(device)

            def __getitem__(self, index):
                return (self._data[:,index * DatasetCreator.DATAFRAME_LENGTH: (index + 1) * DatasetCreator.DATAFRAME_LENGTH],self._label[index])


            def __len__(self):
                return len(self._label)


class Sampler():

    def __init__(self, dataset : _InnerDataset, workers : int, sampler: Sampler, batch_size):
      self._sampler = sampler
      self._dataset = dataset
      self._batch_size = batch_size
      self._workers = workers
      self.sampler = sampler

    def __iter__(self):
        if (len(self._dataset) == 0):
            raise Exception("you can't iterate empty dataset")
        self._loader =  iter(DataLoader(self._dataset,batch_size= self._batch_size, sampler=self.sampler(data_source=self._dataset), num_workers=self._workers))
        return self

    def __next__(self):
        tensor = torch.tensor(np.empty((self._batch_size, len(DatasetCreator.CHOSEN_COLUMNS), DatasetCreator.DATAFRAME_LENGTH)))
        label = torch.tensor(np.empty(self._batch_size))
        for i in range(self._batch_size):
            tensor[i][:][:], label[i] =  next(self._loader)
        return tensor, label

class DatasetCreator():
    """
    Class for loading data and doing traint-test split from dataset. It also creates loaders for evaluating and testing.
    """

    CSV_HEADER= ["loss","f2","acc","PR_AUC","ROC"]
    CHOSEN_COLUMNS = ["temperature","s_xx","s_xy","s_xz","s_yy","s_yz","s_zz","f_a"]
    DATAFRAME_LENGTH = 25 #need to be divisible by the 500
    @staticmethod
    def _read_dataset(dataset):
        stages = []
        labels = []
        for file, label in DatasetCreator._read_files(dataset):
            parts = int(file.shape[0]/DatasetCreator.DATAFRAME_LENGTH)
            stages.extend(np.array_split(file,parts))
            labels.extend([label] * parts)
        return stages, labels

    @staticmethod
    def _read_files(dataset):
        for cur_dir in sorted(os.scandir(dataset),key= lambda dir: dir.name):
                if not cur_dir.name.startswith('.') and cur_dir.is_dir():
                    length = float(cur_dir.path.split('/')[-1])
                    for dpd in sorted(os.scandir(os.path.join(cur_dir)),key= lambda dir: dir.name):
                        block1 = float(dpd.path.split('/')[-1].split('_')[0])
                        block2 = float(dpd.path.split('/')[-1].split('_')[1])
                        stages = 0
                        time = 0.0
                        timestep = 0.0
                        for sim in sorted(os.scandir(os.path.join(dpd)),key= lambda dir: dir.name):
                            label = 0.0
                            if os.path.isfile(os.path.join(sim,'1')):
                                label=1.0
                            try:
                                with open(os.path.join(sim,"CORREL")) as f:
                                    stages +=1
                                    lines = f.readlines()
                                    #the times is cumulative, so we only take the value from the second row and that use it as step
                                    timestep = float(lines[2].split(" ")[2])
                                    values ={"time": [], "en_total": [], "pe_total": [], "be_total": [], "pressure": [], "s_xx": [], "s_xy": [], "s_xz": [], "s_yx": [], "s_yy": [],
                                     "s_yz": [], "s_zx": [],"s_zy": [], "s_zz": [], "temperature": [], "bndlen_av": [], "bndlen_max": [], "bndlen_min": [], "len": [],
                                     "f_a": []}
                                    for row in lines[1:]:
                                        #append the values which are the same for all lines in the file
                                        values["len"].append(length)
                                        values["f_a"].append(block1/(block1+block2))
                                        values["time"].append(time)

                                        time += timestep
                                        dict_iter= iter(list(values.keys())[1:-1])
                                        for val in iter(row.split(" ")[4:]):
                                            if val:
                                                values[next(dict_iter)].append(float(val))
                                    yield pd.DataFrame(values), label
                            except:
                                None
        #standardize dataset
        # std = StandardScaler().fit_transform(pd.DataFrame(values).to_numpy())
        # return pd.DataFrame(std, columns=values.keys()).join(pd.DataFrame({"labels":labels}))

    def __init__(self,dataset,load=False, workers=0, device="cpu"):
        """

        Initialization loads the data into memory

        Args:
            dataset (string): path to the root of dataset
            load (bool, optional): If load indexs from file. Defaults to None.
        """
        from transformation import transform_data

        
        self._workers = workers
        self._device = device
        stages, labels = self._read_dataset(dataset)
        #self.data = transform_data(self.data)
        if load:
            with open("index_split.json", "r") as f:
                dict = json.load(f)
            dataframe_train = dict["train"]
            dataframe_test= dict["test"]
        else:
            stages_indexes = range(len(stages))
            dataframe_train, dataframe_test = train_test_split(stages_indexes, train_size=0.8, test_size=0.2, stratify=labels)
            ''' with open("index_split.json", "w") as f:
                json.dump({"train": dataframe_train, "test":dataframe_test},f)'''
        self._dataframe_train = _InnerDataset(pd.concat([stages[i] for i in dataframe_train]), np.array([labels[i] for i in dataframe_train]))
        self._dataframe_test = _InnerDataset(pd.concat([stages[i] for i in dataframe_test]), np.array([labels[i] for i in dataframe_test]))
        

        return


    def loader_train(self, batch_size=64):
        """
        Create loader which is suppose to be used for training

        Args:
            batch_size (int, optional): size of the batches. Defaults to 64.

        Returns:
            iter: iterator which will create batches of data.
        """
        return DataLoader(self._dataframe_train,batch_size= batch_size, sampler=RandomSampler(data_source=self._dataframe_train), num_workers=self._workers) #FIXME: implement worker variable


    def loader_evaluate_train(self, batch_size=64):
        """
        Create loader which is supposed to be used for evaluation of train dataset.

        Args:
            batch_size (int, optional): size of the batches. Defaults to 64.

        Returns:
            iter: iterator which will create batches of data.
        """
        return DataLoader(self._dataframe_train,batch_size= batch_size, sampler=SequentialSampler(data_source=self._dataframe_train), num_workers=self._workers) 

    def loader_evaluate_test(self, batch_size=64):
        """
        Create loader which is supposed to be used for evaluation of test dataset

        Args:
            batch_size (int, optional): size of the batches. Defaults to 64.

        Returns:
            iter: iterator which will create batches of data.
        """
        return DataLoader(self._dataframe_test, batch_size= batch_size, sampler=SequentialSampler(data_source=self._dataframe_test), num_workers=self._workers) 