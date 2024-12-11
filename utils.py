import torch
import torch_geometric as pyg
import scipy
import pickle
import os
import datetime
import tqdm
from torch.utils.tensorboard import SummaryWriter


def read_deezer_data():
    data = None
    if not os.path.exists('data/preprocessed_deezer_data.pkl'):
        # Run `python preprocess_deezer_data.py` first
        print('Deezer data not found')
        exit(1)
    with open('data/preprocessed_deezer_data.pkl', 'rb') as f:
        data, alter_edges, edge_attr = pickle.load(f)
    return data, alter_edges, edge_attr

def read_reddit_data():
    data = None
    if not os.path.exists('data/preprocessed_reddit_title_data.pkl'):
        # Run `python preprocess_reddit_data.py` first
        print('Reddit data not found')
        exit(1)
    with open('data/preprocessed_reddit_title_data.pkl', 'rb') as f:
        data, alter_edges, edge_attr, _ = pickle.load(f)
    return data, alter_edges, edge_attr

def print_performance(performance, data_name, model_name, epoch=None):
    if epoch is not None:
        print('Performance epoch {} of {} on {} dataset'.format(epoch, model_name, data_name))
    else:
        print('Performance of {} on {} dataset'.format(model_name, data_name))
    for key, value in performance.items():
        print('{}: {}'.format(key, value))

def load_performance(data_name, model_name, epoch=None):
    if epoch is None:
        with open('performance/performance_{}_{}.pkl'.format(data_name, model_name), 'rb') as f:
            return pickle.load(f)
    else:
        with open('cache_performance/performance_{}_{}_epoch_{}.pkl'.format(data_name, model_name, epoch), 'rb') as f:
            return pickle.load(f)

def save_performance(performance, data_name, model_name, epoch=None):
    if epoch is None:
        if not os.path.exists('performance'):
            os.makedirs('performance')
        with open('performance/performance_{}_{}.pkl'.format(data_name, model_name), 'wb') as f:
            pickle.dump(performance, f)
    else:
        if not os.path.exists('cache_performance'):
            os.makedirs('cache_performance')
        with open('cache_performance/performance_{}_{}_epoch_{}.pkl'.format(data_name, model_name, epoch), 'wb') as f:
            pickle.dump(performance, f)

def save_model(model, data_name, model_name, epoch=None):
    if epoch is None:
        if not os.path.exists('models'):
            os.makedirs('models')
        torch.save(model.state_dict(), 'models/model_{}_{}.pt'.format(data_name, model_name))
    else:
        if not os.path.exists('cache_models'):
            os.makedirs('cache_models')
        torch.save(model.state_dict(), 'cache_models/model_{}_{}_epoch_{}.pt'.format(data_name, model_name, epoch))

def load_model(data_name, model_name, epoch=None):
    if epoch is None:
        model = CustomModel(model_name, data_name)
        model.load_state_dict(torch.load('models/model_{}_{}.pt'.format(data_name, model_name), weights_only=True))
        return model
    else:
        model = CustomModel(model_name, data_name)
        model.load_state_dict(torch.load('cache_models/model_{}_{}_epoch_{}.pt'.format(data_name, model_name, epoch)))
        return model
    
class CustomModel(torch.nn.Module):
    def __init__(self, model_name, data_name):
        super(CustomModel, self).__init__()
        self.model_name = model_name
        if data_name == 'deezer':
            self.out_features = 1
            self.hidden_features = 64
            self.d2 = 3
        elif data_name == 'reddit':
            self.out_features = 300
            self.hidden_features = 512
            self.d2 = 6
        self.conv1 = pyg.nn.GCNConv(1, self.hidden_features, normalize=False)
        self.batchnorm1a = torch.nn.BatchNorm1d(self.hidden_features)
        self.alters1a = torch.nn.Linear(self.d2, self.hidden_features)
        self.alters1b = torch.nn.Linear(self.hidden_features, self.hidden_features)
        self.batchnorm1b = torch.nn.BatchNorm1d(self.hidden_features)
        self.conv2 = pyg.nn.GCNConv(self.hidden_features, self.hidden_features, normalize=False)
        self.batchnorm2a = torch.nn.BatchNorm1d(self.hidden_features)
        self.alters2a = torch.nn.Linear(self.d2, self.hidden_features)
        self.alters2b = torch.nn.Linear(self.hidden_features, self.hidden_features)
        self.batchnorm2b = torch.nn.BatchNorm1d(self.hidden_features)
        self.linear = torch.nn.Linear(self.hidden_features, self.out_features)

        self.alter_edge_attr = None
    
    def forward(self, x, edge_index, alter_edges=None, edge_attr=None):
        x = self.conv1(x, edge_index)
        x = self.batchnorm1a(x)
        if self.model_name != 'baseline':
            assert alter_edges is not None
            assert edge_attr is not None
            if self.alter_edge_attr is None:
                self.alter_edge_attr = torch.t(torch.sparse.mm(edge_attr, alter_edges))
            x += self.batchnorm1b(self.alters1b(self.alters1a(self.alter_edge_attr)))
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = self.batchnorm2a(x)
        if self.model_name != 'baseline':
            x += self.batchnorm2b(self.alters2b(self.alters2a(self.alter_edge_attr)))
        x = torch.relu(x)
        x = self.linear(x)
        return x

def train_model(model, data, data_name, model_name, alter_edges=None, edge_attr=None):
    # Print current date and time
    print('Start training...')
    print(datetime.datetime.now())
    # Start training
    # Use GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
        data = data.to('cuda')
        if alter_edges is not None:
            alter_edges = alter_edges.to('cuda')
            edge_attr = edge_attr.to('cuda')
    # Create nan mask (if the first dimension of data.y is nan, then the masking value will be False)
    nan_mask = ~torch.isnan(data.y[:, 0])
    model.train()
    if data_name == 'deezer':
        LR = 0.001
        N_EPOCHS = 1000
    elif data_name == 'reddit':
        LR = 0.002
        N_EPOCHS = 1000
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    if data_name == 'deezer':
        loss = torch.nn.MSELoss()
    elif data_name == 'reddit':
        loss = torch.nn.CosineEmbeddingLoss()
    writer = SummaryWriter(comment='_{}_{}'.format(data_name, model_name))
    for epoch in tqdm.tqdm(range(N_EPOCHS)):
        optimizer.zero_grad()
        if model_name == 'baseline':
            out = model(data.x, data.edge_index)
        else:
            out = model(data.x, data.edge_index, alter_edges, edge_attr)
        if data_name == 'deezer':
            loss_value = loss(out[nan_mask & data.train_mask], data.y[nan_mask & data.train_mask])
        elif data_name == 'reddit':
            loss_value = loss(out[nan_mask & data.train_mask], data.y[nan_mask & data.train_mask], torch.ones(sum(nan_mask & data.train_mask)))
        # Log in tensorboard
        writer.add_scalar('Loss/train', loss_value.item(), epoch)

        loss_value.backward()
        optimizer.step()
        if epoch % 10 == 0:
            # print('Epoch {}: Loss {}'.format(epoch, loss_value.item()))
            if model_name == 'baseline':
                performance = test_model(model, data, data_name, mode='valid')
            else:
                performance = test_model(model, data, data_name, alter_edges, edge_attr, mode='valid')
            # print_performance(performance, data_name, model_name, epoch)
            save_performance(performance, data_name, model_name, epoch)
            save_model(model, data_name, model_name, epoch)
            if data_name == 'deezer':
                writer.add_scalar('MSE/valid', performance['MSE'], epoch)
                writer.add_scalar('MAE/valid', performance['MAE'], epoch)
                writer.add_scalar('MSE_std/valid', performance['MSE_std'], epoch)
                writer.add_scalar('MAE_std/valid', performance['MAE_std'], epoch)
            else:
                writer.add_scalar('Cosine similarity/valid', performance['Cosine similarity'], epoch)
                writer.add_scalar('Cosine similarity std/valid', performance['Cosine similarity std'], epoch)
            writer.flush()

    writer.close()
    print('Done training!')
    print(datetime.datetime.now())
    return model

def test_model(model, data, data_name, alter_edges=None, edge_attr=None, mode='test'):
    performance = {}
    model.eval()
    mask = data.test_mask if mode == 'test' else data.val_mask
    nan_mask = ~torch.isnan(data.y[:, 0])
    with torch.no_grad():
        if model.model_name == 'baseline':
            out = model(data.x, data.edge_index)
        else:
            out = model(data.x, data.edge_index, alter_edges, edge_attr)
        if data_name == 'deezer':
            # Mean squared error & mean absolute error
            performance['MSE'] = torch.nn.MSELoss()(out[nan_mask & mask], data.y[nan_mask & mask]).item()
            performance['MAE'] = torch.nn.L1Loss()(out[nan_mask & mask], data.y[nan_mask & mask]).item()
            # Also the standard deviation of the two types of errors
            performance['MSE_std'] = torch.std((out[nan_mask & mask] - data.y[nan_mask & mask]) ** 2).item()
            performance['MAE_std'] = torch.std(torch.abs(out[nan_mask & mask] - data.y[nan_mask & mask])).item()
        elif data_name == 'reddit':
            # Consider the cosine similarity between the predicted and true embeddings
            performance['Cosine similarity'] = torch.nn.CosineSimilarity()(out[nan_mask & mask], data.y[nan_mask & mask]).mean().item()
            # Also the standard deviation of the cosine similarity
            performance['Cosine similarity std'] = torch.std(torch.nn.CosineSimilarity()(out[nan_mask & mask], data.y[nan_mask & mask])).item()
    return performance
