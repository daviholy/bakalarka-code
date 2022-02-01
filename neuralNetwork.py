
from torch import nn,round
from torchsummary import summary
import torch, csv, os
import numpy as np
from torch.optim import Adam
from datasetCreator import DatasetCreator
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.metrics import fbeta_score

class NeuralNetwork(nn.Module):
    
    def __init__(self,num_of_features, device ="cpu", neurons_per_layer=300, hidden_layers=5, hidden_activations = nn.LeakyReLU) -> None:
        super().__init__()
        self.device = device
        self.hidden_layers = hidden_layers
        self.input_layer = nn.Sequential(nn.BatchNorm1d(num_of_features).to(device), nn.Linear(num_of_features,neurons_per_layer).to(device))
        for i in range(hidden_layers):
            setattr(self,f"hidden_layer_{i}",nn.Sequential(nn.Linear(neurons_per_layer, neurons_per_layer).to(device), nn.BatchNorm1d(neurons_per_layer).to(device), hidden_activations().to(device)))
        self.layer_output = nn.Sequential(nn.Linear(neurons_per_layer, 1).to(device))
        self.return_sigmoid = nn.Sigmoid().to(device)


    def forward(self, x):
        x = self.input_layer(x)
        for i in range(self.hidden_layers):
            x = getattr(self,f"hidden_layer_{i}")(x)
        x = self.layer_output(x).squeeze(1)
        return(self.return_sigmoid(x) if self.training else x)


    def validation(self,loader, file=None, write= False, Print=False):

        binary_threshold = 0.4
            
        def acc(labels_predicted,labels_true):
            correct = (round(labels_predicted) == labels_true).sum()
            return (correct/labels_true.shape[0] * 100).item()

        with torch.no_grad():
            criterion = nn.BCEWithLogitsLoss()
            loss_epoch=0
            labels_dataset = torch.tensor(np.empty(len(loader.dataset))).to(self.device)
            output_dataset = torch.tensor(np.empty(len(loader.dataset))).to(self.device)
            for i, (data,labels) in enumerate(loader):
                output = self(data.float())
                index = i * loader.batch_size
                labels_dataset[index : index + len(labels)] = labels
                output_dataset[index : index + len(labels)] = output

                loss_epoch+=criterion(output,labels).item()

        output_dataset = self.return_sigmoid(output_dataset)
        res = {"loss": loss_epoch/len(loader), "f2": fbeta_score(labels_dataset.detach().numpy(),(output_dataset > binary_threshold).detach().numpy(), beta=2),"acc": acc(output_dataset , labels_dataset), 
        "PR_AUC": average_precision_score(labels_dataset.detach().numpy(), output_dataset.detach().numpy()), "ROC" : roc_auc_score(labels_dataset.cpu().numpy(), output_dataset.detach().numpy())}

        if Print:
            print(f'  loss: {res["loss"]:.4f} | F2 : {res["f2"]:.3f} | acc: {res["acc"]:.3f} | PR_AUC: {res["PR_AUC"]:.3f} | ROC: {res["ROC"]:.3f}')
        if write:
            with open(file,"a") as f:
                writer = csv.DictWriter(f, DatasetCreator.CSV_HEADER)
                writer.writerow(res)
        return res


    def train_model(self, datasetCreator, epochs = 5, learning_rate=0.02, batch_size=64 ,save="./"):

        


        def check_files(train_file,test_file):
            if not os.path.isfile(train_file):
                with open(train_file,"a")as f:
                    header= csv.DictWriter(f,DatasetCreator.CSV_HEADER)
                    header.writeheader()

            if not os.path.isfile(test_file):
                with open(test_file,"a")as f:
                    header= csv.DictWriter(f,DatasetCreator.CSV_HEADER)
                    header.writeheader()


        def check_grad(parameters):
            for list in parameters:
                for grad in list.grad:
                    if  grad.dim() > 0:
                        for value in grad:
                            if value > float("1e5"):
                                print (f"exploding gradient: {grad}")
                    else:
                        if grad > float("1e5"):
                                print (f"exploding gradient: {grad}")


        check_files(f"{save}train.csv", f"{save}test.csv")
        criterion = nn.BCEWithLogitsLoss()
        optimizer = Adam(self.parameters(), lr=learning_rate)
        train_loader = datasetCreator.loader_train(batch_size=batch_size)
        train_eval_loader = datasetCreator.loader_evaluate_train(batch_size=batch_size)
        test_eval_loader = datasetCreator.loader_evaluate_test(batch_size=batch_size)

        self.train()

        for epoch in range(epochs):
            loss_epoch = 0
            #train
            for i,(data,labels) in enumerate(train_loader):
                optimizer.zero_grad()

                output = self(data.float())

                loss = criterion(output,labels)
                loss.backward()
              #  check_grad(self.parameters())
                optimizer.step()

                loss_epoch += loss.item()

                if i%100 ==0:
                    print(f"{i}/{len(train_loader)}")

            #evaluation
            print(f"Epoch: {epoch+1+0:03}")
            print("train")
            self.validation(train_eval_loader,f"{save}train.csv",Print=True,write=True)
            print("test")
            self.validation(test_eval_loader,f"{save}test.csv",Print=True,write=True)
            print()
        return


