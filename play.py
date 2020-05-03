import torch

from utils.viewer import Viewer
from teeny_go.policy_network import PolicyNetwork
from teeny_go.value_network import ValueNetwork

value_net = ValueNetwork(alpha=0.00001, num_res=12, num_channel=256)
sd = torch.load("VN-R12-C256-V9.pt", map_location={'cuda:0': 'cpu'})
value_net.load_state_dict(state_dict=sd)

# load and set up netowork
policy_net = PolicyNetwork(alpha=0.00001, num_res=12, num_channel=256)
sd = torch.load("PN-R12-C256-P9.pt", map_location={'cuda:0': 'cpu'})
policy_net.load_state_dict(state_dict=sd)

viewer = Viewer()

viewer.human_vs_ai(policy_net, value_net)
