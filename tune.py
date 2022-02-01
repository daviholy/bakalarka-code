import os
from ray.tune.schedulers import HyperBandForBOHB
from ray.tune.suggest.bohb import TuneBOHB
from ray import tune
from neuralNetwork import NeuralNetwork
from torch import nn,save,load
from torch.optim import Adam
from ray.tune.suggest import ConcurrencyLimiter
import torch,argparse
from datasetCreator import DatasetCreator
from ray.tune.schedulers.pb2 import PB2

class Trainable(tune.Trainable):
    def setup(self, config):
        self.lr = config["lr"]
        self.nn = NeuralNetwork(8,device=config["device"], neurons_per_layer=config["neurons_per_layer"], hidden_layers=config["hidden_layers"], hidden_activations= config["activation"])
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = Adam(self.nn.parameters(), lr=config["lr"])
        self.loader_train = config["dataset"].loader_train(batch_size=config["batch"])
        self.loader_evaluate = config["dataset"].loader_evaluate_test(batch_size=config["batch"])

    def step(self):
        loss_epoch = 0
        
        for i ,(data,labels) in enumerate(self.loader_train):
            self.nn.train()
            self.optimizer.zero_grad()

            output = self.nn(data.float())

            loss = self.criterion(output,labels)
            loss.backward()

            self.optimizer.step()
            loss_epoch += loss.item()

            #acc=self.nn.validation(self.loader_evaluate)

            if i%100 ==0:
                print(f"{i}/{len(self.loader_train)}")

        return{"loss": loss_epoch/len(self.loader_train)}

    def save_checkpoint(self,tmp_checkpoint_dir):
        save(self.nn.state_dict(),os.path.join(tmp_checkpoint_dir,'model.pt'))
        return tmp_checkpoint_dir

    def load_checkpoint(self, checkpoint):
        dir = os.path.join(checkpoint, "model.pt")
        self.nn.load_state_dict(torch.load(dir))
        
if __name__ == "__main__":

    def test(dataset,device):
        config = {"device": device, "hidden_layers": 5, "lr": 0.02, "neurons_per_layer": 300, "batch": 64, "dataset": dataset}
        train = Trainable(config)
        train.step()

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory', type=str, default='./dataset',
                        help='directory where the dataset is stored')
    parser.add_argument('-w', '--workers', type=int, default=4,
                        help='num of workers for fetching data')

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    dataset = DatasetCreator(args.directory, workers=args.workers, device=device)

    test(dataset,device)

    # scheduler= PB2(metric="loss", mode="min", hyperparam_bounds = {
    #     "lr":[2e-1, 2e-5], "batch": [8, 256]},
    #     perturbation_interval=15)

    algo= TuneBOHB(metric="loss", mode="min")
    bohb = HyperBandForBOHB( time_attr="training_iteration", metric="loss", mode="min", max_t = 100)
    algo = ConcurrencyLimiter(algo,4)

    config = {
        "device":device, "lr": tune.loguniform(1e-4, 1e-1), 
        "neurons_per_layer": tune.choice([8,16,32,64,128]), 
        "hidden_layers":tune.choice([2,3,4,5,6]), "batch": tune.choice([16,32,64,128,256]),
        "activation": tune.choice([nn.LeakyReLU, nn.PReLU, nn.Hardswish, nn.ELU]), 
        "dataset": dataset}
    analysis = tune.run(Trainable, name="Bayes", local_dir="/home/dave/ray", sync_config=tune.SyncConfig(syncer=None),
     num_samples=42, scheduler=bohb, search_alg=algo,
        config = config)

    print(f'Best hyperparameters found were: {analysis.get_best_config(mode="min",metric="loss")}')