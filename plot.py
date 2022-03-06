import matplotlib
matplotlib.use("TkAgg")

import seaborn as sns
import matplotlib.pyplot as plt
import argparse, csv
from datasetCreator import DatasetCreator


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="My Script")
    parser.add_argument('-d', '--directory', type=str, default='./dataset',
                        help='directory where the dataset is stored')
    parser.add_argument('-t',"--plot-training", action='store_true', 
        help="plot the graphs from data generated during training")
    parser.add_argument('-mat',"--plot-matrix", action='store_true',
        help="plot the scatterplot matrix for dataset (it uses matplotlib for now, which can be really SLOW if you try to plot a lot of points)")
    parser.add_argument('-s',"--store-dataset", default=None, help="path, where to store image of scatter matrix from dataset")
    parser.add_argument('-d3', default=None,  nargs=3, help="do a 3d plot of given names of dimensions")
    args = parser.parse_args()


    if args.plot_training:
        with open("./train.csv") as f:
            reader = csv.DictReader(f)
            train = {}
            for x in reader.fieldnames : train[x] = []
            for line in reader:
                for x in reader.fieldnames : train[x].append(float(line[x]))

        with open("./test.csv") as f:
            reader =csv.DictReader(f)
            test = dict.fromkeys(reader.fieldnames,[])
            for line in reader:
                for x in reader.fieldnames : test[x].append(float(line[x]))

        for key in train.keys():
            plt.plot(train[key])
            plt.xlabel("epoch")
            plt.ylabel(key)
            plt.show()
            #TODO: implement saving into folder instead of interactiveli showing a lot of windows

    if args.d3:
        import plotly.graph_objects as go
        dataset = DatasetCreator(args.directory)
        plot= go.Figure( data= go.Scatter3d(
            x=dataset.data[args.d3[0]], 
            y=dataset.data[args.d3[1]], 
            z=dataset.data[args.d3[2]],
            mode='markers',
            marker={
                "size":2.5,
                "color": dataset.data['label'],
                "opacity":1,
                "line" : { "width" : 0}
            }
        ))
        plot.show()

    if args.plot_matrix:
        plt.switch_backend('Agg')
        data = DatasetCreator.read_dataset(args.directory)
        data[0]["label"] = data[1]
        plt = sns.pairplot(data[0], hue='label', kind='hist', palette={0:sns.color_palette("Paired")[0],1:sns.color_palette("Paired")[5]})
        plt.savefig("./matrix.png")
    