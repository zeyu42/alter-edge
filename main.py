import torch
import torch_geometric as pyg
from utils import *
import sys

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python main.py <dataset> <model>')
        sys.exit(1)
    data_name = sys.argv[1]
    model_name = sys.argv[2]
    if data_name == 'deezer':
        data, alter_edges, edge_attr = read_deezer_data()
    elif data_name == 'reddit':
        data, alter_edges, edge_attr = read_reddit_data()
    else:
        print('Unknown dataset')
        print('Available datasets: deezer, reddit')
        sys.exit(1)

    if model_name in ['baseline', 'ego', 'alters']:
        model = CustomModel(model_name=model_name, data_name=data_name)
    else:
        print('Unknown model')
        print('Available models: baseline, ego, alters')
        sys.exit(1)
    if model_name == 'baseline':
        model = train_model(model, data, data_name, model_name)
        performance = test_model(model, data, data_name)
    else:
        if model_name == 'ego':
            edge_attr[0:edge_attr.shape[0]//3,] = 0 # To fix the parameter size, set the first 1/3 to zero rather than truncate it
        model = train_model(model, data, data_name, model_name, alter_edges, edge_attr)
        performance = test_model(model, data, data_name, alter_edges, edge_attr)
    print_performance(performance, data_name, model_name)
    save_performance(performance, data_name, model_name)
    save_model(model, data_name, model_name)
