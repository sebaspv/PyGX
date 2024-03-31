import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torch_geometric.datasets import ZINC
from torch_geometric.nn import VGAE
import torch_geometric.transforms as T
from torch_geometric.utils import train_test_split_edges
from encoders import VGCNEncoder

dataset = ZINC("data/", subset=True)
data = dataset[0]
data.train_mask = data.val_mask = data.test_mask = data.y = None
data = train_test_split_edges(data)
writer = SummaryWriter("runs/VGAE_experiment_" + "2d_100_epochs")

out_channels = 16
n_features = data.num_features
epochs = 300

model = VGAE(VGCNEncoder(n_features, out_channels))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
data.x = F.normalize(data.x.float())
x = data.x.to(device)
train_pos_edge_index = data.train_pos_edge_index.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train():
    model.train()
    optimizer.zero_grad()
    z = model.encode(x, train_pos_edge_index)
    loss = model.recon_loss(z, train_pos_edge_index)

    loss = loss + (1 / data.num_nodes) * model.kl_loss()
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test(pos_edge_index, neg_edge_index):
    model.eval()
    z = model.encode(x, train_pos_edge_index)
    return model.test(z, pos_edge_index, neg_edge_index)


for epoch in range(1, epochs + 1):
    loss = train()
    auc, ap = test(data.test_pos_edge_index, data.test_neg_edge_index)
    writer.add_scalar("auc train", auc, epoch)
    writer.add_scalar("ap train", ap, epoch)
    print(f"Epoch: {epoch} AUC: {auc} AP: {ap}")
