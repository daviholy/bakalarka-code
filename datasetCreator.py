import os
from torch.utils.data import Dataset,RandomSampler, SequentialSampler
from torch.utils.data.dataloader import DataLoader

DATASET="/home/david/skola/bakalarka-fortran/new_dataset/"

class DatasetCreator():
    """
    Class for loading data and doing traint-test split from dataset. It also creates loaders for evaluating and testing.

    Args:
        Dataset (string): [description]
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
            return (self._data["time"][index], self._data["en_total"][index], self._data["pe_total"][index], self._data["be_total"][index],
             self._data["pressure"][index], self._data["s_xx"][index], self._data["s_xy"][index], self._data["s_xz"][index],
             self._data["s_yx"][index], self._data["s_yy"][index], self._data["s_yz"][index], self._data["s_zx"][index], self._data["s_zy"][index], 
             self._data["s_zz"][index], self._data["temperature"][index], self._data["bndlen_av"][index], self._data["bndlen_max"][index], self._data["bndlen_min"][index])


    def __init__(self,dataset,train=None):
        """

        Initialization loads the data into memory

        Args:
            dataset (string): path to the root of dataset
            train (string, optional): path to the file which specifies the indexes which are meant to be in train dataset. If None it will split randomly stratified. Defaults to None.
        """
        #FIXME: just placeholder now split and store separately into train-test later
        self._values ={"time": [], "en_total": [], "pe_total": [], "be_total": [], "pressure": [], "s_xx": [], "s_xy": [], "s_xz": [], "s_yx": [], "s_yy": [],
         "s_yz": [], "s_zx": [],"s_zy": [], "s_zz": [], "temperature": [], "bndlen_av": [], "bndlen_max": [], "bndlen_min": []} 
        for cur_dir in os.listdir(dataset):
            if os.path.isdir(os.path.join(dataset,cur_dir)) and cur_dir != ".git":
                for (root,dirs,files) in os.walk(os.path.join(dataset,cur_dir)):
                    try:
                        with open(os.path.join(root,"CORREL"),"r") as f:
                            for row in f.readlines()[1:]:
                                dict_iter= iter(self._values)
                                for val in iter(row.split(" ")):
                                        if val:
                                            self._values[next(dict_iter)].append(float(val))
                    except:
                        None
        #check if all fields have same length
        assert len(self._values["time"]) == len(self._values["en_total"]) == len(self._values["pe_total"]) == len(self._values["be_total"]) == len(self._values["pressure"]) \
            == len(self._values["s_xx"]) == len(self._values["s_xy"]) == len(self._values["s_xz"]) == len(self._values["s_yx"]) == len(self._values["s_yy"]) == len(self._values["s_yz"]) \
            == len(self._values["s_zx"]) == len(self._values["s_zy"]) == len(self._values["s_zz"]) == len(self._values["temperature"]) == len(self._values["bndlen_av"]) == len(self._values["bndlen_max"]) \
            == len(self._values["bndlen_min"])
            #TODO: implement train-test split
        if train:
            #TODO: implement loading train set with indexes loaded from file specified in train variable
            None
        else:
            #TODO: create fresh splitting
            None
        return
        
    def loader_train(self,batch_size=64):
        """
        Create loader which is suppose to be used for training

        Args:
            batch_size (int, optional): size of the batches. Defaults to 64.

        Returns:
            iter: iterator which will create batches of data.
        """

        return DataLoader(self._values,batch_size= batch_size, sampler=RandomSampler ) # FIXME: add actual train dataset 

    def iterator_evaluate_train(self, batch_size=64):
        """
        Create loader which is supposed to be used for evaluation of train dataset.

        Args:
            batch_size (int, optional): size of the batches. Defaults to 64.

        Returns:
            iter: iterator which will create batches of data.
        """

        return DataLoader(self._values,batch_size= batch_size, sampler=SequentialSampler) # FIXME: add actual train dataset

    def iterator_evaluate_test(self,batch_size=64):
        """
        Create loader which is supposed to be used for evaluation of test dataset

        Args:
            batch_size (int, optional): size of the batches. Defaults to 64.

        Returns:
            iter: iterator which will create batches of data.
        """

        return DataLoader(self._values, batch_size= batch_size, sampler=SequentialSampler) # FIXME: add actual test dataset


if __name__ == "__main__":
    dataset = DatasetCreator(DATASET)
    None