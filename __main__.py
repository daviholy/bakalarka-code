from datasetCreator import DatasetCreator
from neuralNetwork import NeuralNetwork
from torch import save
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
                        help="specify number of epochs")
    parser.add_argument('-l', '--lr', type=float, default=0.02,
                        help="specify learning rate") 
                           
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    dataset = DatasetCreator(args.directory, workers=args.workers, batch_size=args.batchsize)
    print("data loaded")
    model = NeuralNetwork(device=device)
    model.train_model(dataset, save=args.save, epochs=args.epochs, learning_rate=args.lr)
    save(model.state_dict(), "model.pth")