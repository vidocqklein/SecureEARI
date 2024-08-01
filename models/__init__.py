from models.Elastic_gnn.elasticgnn import ElasticGNN
from models.GCN import GCN, surrogate_GCN
from models.mlp import MLP
from models.EARImodel.EARI import EARI

model_map = {
    'gcn': GCN,
    'mlp': MLP,
    'eari': EARI,
    'surrogate_GCN': surrogate_GCN,
}