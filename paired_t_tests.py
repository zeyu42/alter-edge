import torch
from scipy.stats import ttest_rel
from utils import read_deezer_data, read_reddit_data, load_model, print_performance, test_model

print('Comparing deezer results')
data, alter_edges, edge_attr = read_deezer_data()
baseline_model = load_model('deezer', 'baseline')
ego_model = load_model('deezer', 'ego')
alters_model = load_model('deezer', 'alters')
models = [baseline_model, ego_model, alters_model]

nan_mask = ~torch.isnan(data.y[:, 0])
print('Test size:', data.y[nan_mask & data.test_mask].shape[0])
MSEs = []
MAEs = []
with torch.no_grad():
    for model in models:
        model.eval()
        if model.model_name == 'baseline':
            out = model(data.x, data.edge_index)
        elif model.model_name == 'ego':
            edge_attr_ego = edge_attr.detach().clone()
            edge_attr_ego[:edge_attr_ego.shape[0]//3] = 0
            out = model(data.x, data.edge_index, alter_edges, edge_attr_ego)
        else:
            out = model(data.x, data.edge_index, alter_edges, edge_attr)
        pred = out[nan_mask & data.test_mask]
        MSE = torch.nn.MSELoss(reduction='none')(pred, data.y[nan_mask & data.test_mask])
        MSEs.append(MSE)
        MAE = torch.nn.L1Loss(reduction='none')(pred, data.y[nan_mask & data.test_mask])
        MAEs.append(MAE)
        print(model.model_name, 'mean MSE: {}'.format(MSE.mean().item()))
        print(model.model_name, 'mean MAE: {}'.format(MAE.mean().item()))

# Perform paired t-tests between the model MSEs
print('Comparing MSEs')
t_statistic, p_value = ttest_rel(MSEs[0], MSEs[1])
print('Baseline vs ego model: t_statistic={}, p_value={}'.format(t_statistic, p_value))
t_statistic, p_value = ttest_rel(MSEs[0], MSEs[2])
print('Baseline vs alters model: t_statistic={}, p_value={}'.format(t_statistic, p_value))
t_statistic, p_value = ttest_rel(MSEs[1], MSEs[2])
print('Ego vs alters model: t_statistic={}, p_value={}'.format(t_statistic, p_value))
print('Comparing MAEs')
t_statistic, p_value = ttest_rel(MAEs[0], MAEs[1])
print('Baseline vs ego model: t_statistic={}, p_value={}'.format(t_statistic, p_value))
t_statistic, p_value = ttest_rel(MAEs[0], MAEs[2])
print('Baseline vs alters model: t_statistic={}, p_value={}'.format(t_statistic, p_value))
t_statistic, p_value = ttest_rel(MAEs[1], MAEs[2])
print('Ego vs alters model: t_statistic={}, p_value={}'.format(t_statistic, p_value))


print()
print('Comparing reddit results')
data, alter_edges, edge_attr = read_reddit_data()
baseline_model = load_model('reddit', 'baseline')
ego_model = load_model('reddit', 'ego')
alters_model = load_model('reddit', 'alters')
models = [baseline_model, ego_model, alters_model]

nan_mask = ~torch.isnan(data.y[:, 0])
print('Test size:', data.y[nan_mask & data.test_mask].shape[0])
cosine_similarities = []
with torch.no_grad():
    for model in models:
        model.eval()
        if model.model_name == 'baseline':
            out = model(data.x, data.edge_index)
        elif model.model_name == 'ego':
            edge_attr_ego = edge_attr.detach().clone()
            edge_attr_ego[:edge_attr_ego.shape[0]//3] = 0
            out = model(data.x, data.edge_index, alter_edges, edge_attr_ego)
        else:
            out = model(data.x, data.edge_index, alter_edges, edge_attr)
        pred = out[nan_mask & data.test_mask]
        cosine_similarity = torch.nn.CosineSimilarity()(pred, data.y[nan_mask & data.test_mask])
        cosine_similarities.append(cosine_similarity)
        print(model.model_name, 'mean cosine similarity: {}'.format(cosine_similarity.mean().item()))

# Perform paired t-tests between the model cosine similarities
t_statistic, p_value = ttest_rel(cosine_similarities[0], cosine_similarities[1])
print('Baseline vs ego model: t_statistic={}, p_value={}'.format(t_statistic, p_value))
t_statistic, p_value = ttest_rel(cosine_similarities[0], cosine_similarities[2])
print('Baseline vs alters model: t_statistic={}, p_value={}'.format(t_statistic, p_value))
t_statistic, p_value = ttest_rel(cosine_similarities[1], cosine_similarities[2])
print('Ego vs alters model: t_statistic={}, p_value={}'.format(t_statistic, p_value))

