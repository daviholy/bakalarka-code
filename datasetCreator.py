from sklearn.preprocessing import StandardScaler

import os, json,torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch import tensor
from torch.utils.data import Dataset,RandomSampler, SequentialSampler
from torch.utils.data.dataloader import DataLoader



class DatasetCreator():
    """
    Class for loading data and doing traint-test split from dataset. It also creates loaders for evaluating and testing.
    """

    CSV_HEADER= ["loss","f2","acc","PR_AUC","ROC"]
    @staticmethod
    def read_dataset(dataset):
        values ={"time": [], "en_total": [], "pe_total": [], "be_total": [], "pressure": [], "s_xx": [], "s_xy": [], "s_xz": [], "s_yx": [], "s_yy": [],
         "s_yz": [], "s_zx": [],"s_zy": [], "s_zz": [], "temperature": [], "bndlen_av": [], "bndlen_max": [], "bndlen_min": [], "len": [],
         "block1": [], "block2": []}
        labels= []
        for cur_dir in os.scandir(dataset):
                if not cur_dir.name.startswith('.') and cur_dir.is_dir():
                    leng = float(cur_dir.path.split('/')[-1])
                    for dpd in os.scandir(os.path.join(cur_dir)):
                        block1 = float(dpd.path.split('/')[-1].split('_')[0])
                        block2 = float(dpd.path.split('/')[-1].split('_')[1])
                        time = 0.0
                        timestep = 0.0
                        for sim in sorted(os.listdir(os.path.join(dpd))):
                            if sim != '50':
                                label=0.0
                                if os.path.isfile(os.path.join(dpd,sim,'1')):
                                    label=1.0
                                try:
                                    with open(os.path.join(dpd,sim,"CORREL")) as f:
                                        lines = f.readlines()
                                        #the times is cumulative, so we only take the value from the secodn row and that use it as step
                                        timestep = float(lines[2].split(" ")[2])
                                        for row in lines[1:]:
                                            #append the values which are the same for all lines in the file
                                            labels.append(label)
                                            values["len"].append(leng)
                                            values["block1"].append(block1)
                                            values["block2"].append(block2)

                                            values["time"].append(time)
                                            time += timestep

                                            dict_iter= iter(list(values.keys())[1:-1])
                                            for val in iter(row.split(" ")[4:]):
                                                if val:
                                                    values[next(dict_iter)].append(float(val))
                                except:
                                    None
        #standardize dataset
        # std = StandardScaler().fit_transform(pd.DataFrame(values).to_numpy())
        # return pd.DataFrame(std, columns=values.keys()).join(pd.DataFrame({"labels":labels}))

        return pd.DataFrame(values) , labels

    def __init__(self,dataset,load=False, workers=0, device="cpu"):
        """

        Initialization loads the data into memory

        Args:
            dataset (string): path to the root of dataset
            load (bool, optional): If load indexs from file. Defaults to None.
        """
        from transformation import transform_data
        class _InnerDataset(Dataset):
            """
            Helpfull inner class for creating datasets of splitted data.
            """
            CHOSEN_COLUMNS = ['pe_total_0','pe_total_1','pe_total_2','pe_total_3','pe_total_4','pe_total_5','pe_total_6'
                ,'be_total_0','be_total_1','be_total_2','be_total_3','be_total_4','be_total_5','be_total_6',
                'en_total_0','en_total_1','en_total_2','en_total_3','en_total_4','en_total_5','en_total_6',
                'bndlen_av_0','bndlen_av_1','bndlen_av_2','bndlen_av_3','bndlen_av_4','bndlen_av_5','bndlen_av_6',
                "s_xx","s_yy","s_zz","bndlen_av","pressure","block1","block2"]
            def __init__(self,data, labels):
                """
                initialize the data
                Args:
                    data (dict): dictionary which holds the data
                """
                self._columns = data.shape[1] - 1
                
                self._columns = len(self.CHOSEN_COLUMNS)
                self._data = torch.tensor(data[self.CHOSEN_COLUMNS].values.reshape(-1,).tolist()).to(device)
                self._label = torch.tensor(labels).to(device)

            def __getitem__(self, index):
                return (self._data[self._columns * index : self._columns * (index + 1)],self._label[index])


            def __len__(self):
                return len(self._label)

        
        self._workers = workers
        self._device = device
        self.data, labels = self.read_dataset(dataset)
        self.data = transform_data(self.data)
        self.data= pd.DataFrame(StandardScaler().fit_transform(pd.DataFrame(self.data).to_numpy()),columns=self.data.columns)

        if load:
            with open("index_split.json", "r") as f:
                dict = json.load(f)
            dataframe_train = dict["train"]
            dataframe_test= dict["test"]
        else:
            dataframe_train, dataframe_test = train_test_split(list(range(len(self.data))), train_size=0.8, test_size=0.2, stratify=labels)
            with open("index_split.json", "w") as f:
                json.dump({"train": dataframe_train, "test":dataframe_test},f)
        self._dataframe_train = _InnerDataset(self.data.iloc[dataframe_train], np.array(labels)[dataframe_train])
        self._dataframe_test = _InnerDataset(self.data.iloc[dataframe_test], np.array(labels)[dataframe_test])


        return


    def loader_train(self, batch_size=64):
        """
        Create loader which is suppose to be used for training

        Args:
            batch_size (int, optional): size of the batches. Defaults to 64.

        Returns:
            iter: iterator which will create batches of data.
        """
        return DataLoader(self._dataframe_train,batch_size= batch_size, sampler=RandomSampler(data_source=self._dataframe_train), num_workers=self._workers)


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
