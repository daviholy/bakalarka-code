from torch import nn,round, sigmoid
import torch, csv, os
import numpy as np
from torch.optim import Adam
from sklearn.metrics import average_precision_score


class NeuralNetwork(nn.Module):

    def __init__(self, device ="cpu", l1=36, l2=36, l3=36, l4=36, l5=36) -> None:
        super().__init__()
        self.device = device
        self.layer1 = nn.Sequential(nn.Linear(18,l1, device=device), nn.BatchNorm1d(36, device=device), nn.LeakyReLU())
        self.layer2 = nn.Sequential(nn.Linear(l1, l2, device=device), nn.BatchNorm1d(36, device=device), nn.LeakyReLU())
        self.layer3 = nn.Sequential(nn.Linear(l2, l3, device=device), nn.BatchNorm1d(36, device=device), nn.LeakyReLU())
        self.layer4 = nn.Sequential(nn.Linear(l4, l5, device=device), nn.BatchNorm1d(36, device=device), nn.LeakyReLU())
        self.layer_output = nn.Linear(l5, 1, device=device)


    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.layer_output(x).squeeze(1)

    
    def train_model(self, datasetCreator, epochs = 5, learning_rate=0.02, save="./"):

        CSV_HEADER= ["Acc","PR_AUC"]


        def check_files(train_file,test_file):
            if not os.path.isfile(train_file):
                with open(train_file,"a")as f:
                    header= csv.DictWriter(f, CSV_HEADER)
                    header.writeheader()

            if not os.path.isfile(test_file):
                with open(test_file,"a")as f:
                    header= csv.DictWriter(f, CSV_HEADER)
                    header.writeheader()


        def acc(labels_predicted,labels_true):
            predicted = round(sigmoid(labels_predicted))
            correct = (predicted == labels_true).sum().float()
            return round((correct/labels_true.shape[0]) * 100).cpu().numpy()


        def validation(loader, file):
            with torch.no_grad():
                labels_dataset = torch.tensor(np.empty(len(loader.dataset))).to(self.device)
                output_dataset = torch.tensor(np.empty(len(loader.dataset))).to(self.device)
                for i, (data,labels) in enumerate(loader):
                    data, labels = data.to(self.device), labels.to(self.device)
                    output = self(data.float())
                    index = i * loader.batch_size
                    labels_dataset[index : index + len(labels)] = labels
                    output_dataset[index : index + len(labels)] = output
            res = {"Acc": acc(output_dataset,labels_dataset), "PR_AUC": average_precision_score(round(sigmoid(output_dataset)).cpu().numpy(),labels_dataset.cpu().numpy())}
            print(f'Acc: {res["Acc"]:.3f} | PR_AUC: {res["PR_AUC"]:.3f}')
            with open(file,"a") as f:
                writer = csv.DictWriter(f,CSV_HEADER)
                writer.writerow(res)

        check_files(f"{save}train", f"{save}test")
        criterion = nn.BCEWithLogitsLoss()
        optimizer = Adam(self.parameters(), lr=learning_rate)
        train_loader = datasetCreator.loader_train()
        train_eval_loader = datasetCreator.loader_evaluate_train()
        test_eval_loader = datasetCreator.loader_evaluate_test()

        for epoch in range(epochs):
            loss_epoch = 0
            #train
            self.train()
            for i,(data,labels) in enumerate(train_loader):
                data, labels = data.to(self.device), labels.to(self.device)
                optimizer.zero_grad()

                output = self(data.float())

                loss = criterion(output,labels)
                loss.backward()
                optimizer.step()

                loss_epoch += loss.item()

                if i%100 ==0:
                    print(f"{i}/{len(train_loader)}")

            #evaluation
            self.eval()

            with open(f"{save}loss", "a") as f:
                f.writelines(f"{loss_epoch/len(train_loader)}\n")

            print(f"Epoch {epoch+0:03}")
            print("train")
            print(f"Loss: {loss_epoch/len(train_loader):.5f}")
            validation(train_eval_loader,f"{save}train")
            print("test")
            validation(test_eval_loader,f"{save}test")
            print()
        return


