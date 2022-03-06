from datasetCreator import DatasetCreator
from neuralNetwork import NeuralNetwork
from torchinfo import summary
from torch import save
from os.path import exists
import argparse, torch

DATASET="/home/david/skola/bakalarka-fortran/new_dataset/"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory', type=str, default='./dataset',
                        help='directory where the dataset is stored')
    parser.add_argument('-w', '--workers', type=int, default=0,
                        help='num of workers for fetching data')
    parser.add_argument('-s', '--save', type=str, default='./',
                        help="directory where the statistics will be stored")
    parser.add_argument('-b', '--batchsize', type=int, default=64,
                        help="specify batch size")
    parser.add_argument('-e', '--epochs', type=int, default=5,
                        help="number of epochs")
    parser.add_argument('-l', '--lr', type=float, default=0.002,
                        help="learning rate") 
    parser.add_argument('-hl', '--hidden_layers', type=int, default=4,
                        help="number of hidden layers")
    parser.add_argument('-n', '--neurons_per_layer', type=int, default=4,
                        help="number of neurons")
    parser.add_argument('-sp','--split',type=str,default="index_split.json",
                        help="file name of json file which contains splited indexes of dataset")
    parser.add_argument('-lm', '--load-model', type=str,
                        help="file of saved model to load")
                           
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    load = True if exists(args.split) else False
    print(f"device: {device}")
    dataset = DatasetCreator(args.directory,load=load ,workers=args.workers, device=device)
    print("data loaded\n")

    model = NeuralNetwork(num_of_features= len(DatasetCreator.CHOSEN_COLUMNS),device=device, neurons_per_layer=args.neurons_per_layer, hidden_layers=args.hidden_layers)
    if args.load_model:
        model.load_state_dict(torch.load(args.load_model, map_location=torch.device('cpu')))
    model.to(device)

    model.train_model(dataset, save=args.save, epochs=args.epochs, learning_rate=args.lr, batch_size=args.batchsize)
    save(model.state_dict(), "model.pth")