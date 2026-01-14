from VarUltils.optionsBN import Variables
import copy
from Libraries import *

parser = argparse.ArgumentParser()
parser.add_argument("-num", required=True,help="Number of errors")
parser.add_argument("-T", required=False,help="Target regexp")
parser.add_argument("-r", required=False, action="store_true",help="use to process the second database")
args = parser.parse_args()

device = torch.device("cpu")


# ## Initialize the sample model with sample data
v = Variables()


# # NIR
torch.manual_seed(42)
net = SpikingNet(v)
sample_data = torch.randn(v.num_steps, v.num_inputs)


# Initialize the network with sample data
net.load_state_dict(torch.load("./model/best_rsnn_rl_nav_8.pt",map_location=device))
net = utils.init_network(net, sample_data)


# -------------------------------------------------
# Get the NIR graphv
gp = Graph(net, v.num_inputs)
# Get the NIR model
nir_model = gp.nir_model


# Map neurons to cores
mapping = Mapping(net,nir_model)
total_neurons = mapping.get_total_neurons()
if(args.r==1):
    core_capacity = 25
else:
    core_capacity = max(math.ceil((total_neurons - v.num_outputs) / (v.num_cores - 1)), v.num_outputs)

mapping.set_core_capacity(core_capacity)
mapping.map_neurons()



all_connections =mapping.GetConnections()
nc_multicasts = mapping.GetNcmulticast(all_connections)


# # Create dataset and dataloader
val_set = torch.load('data/test_set_8.pt', weights_only=False)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4, prefetch_factor=2)

# ## Visialization Dataset
print("One batch data shape:", next(iter(val_loader))[0].shape) # (B, T, imputs)
print("One batch label shape:", next(iter(val_loader))[1].shape) # (B)


VisualizeSpikes(val_loader,"./Input_spikes")

net_copy = copy.deepcopy(net).to(device)

# Dictionary to store spikes from each layer
spike_record = {}
hooks = []


reset_spike_record_and_hooks()
# Attach hooks automatically to all Leaky layers
for name, module in net_copy.named_modules():
    if isinstance(module, snn.Leaky) or isinstance(module, snn.RSynaptic):
        hooks.append(module.register_forward_hook(create_spike_hook(name)))

# Get a sample input
data, _ = val_set[0]

# Record spikes
_, _ = net_copy(data.to(device))

# Convert spike records to tensors for easier analysis
for layer_name in spike_record:
    spike_record[layer_name] = torch.squeeze(torch.stack(spike_record[layer_name]))


VisualizeSpikes(spike_record,"./Spike_record")
# Count the number of spikes in each layer
spike_counts = {layer_name: spikes.sum().item() for layer_name, spikes in spike_record.items()}
# -------------------------------------------------

if 'hooks' in globals():
    for hook in hooks:
        hook.remove()
        hooks = []


# Build the lookup table
neuron_lookup = build_neuron_lookup_table(mapping, nc_multicasts)
IdLookup=build_ID_lookup(mapping)



# ## CAM Table
# Generate Instance（use neuron_lookup abd all_connections to generate CAM）
SimpleCAMGenerator(neuron_lookup, all_connections,"./database/CAMTables.sql")
if(args.r==1):
    Name="./Mapping2FaultsInj.sql"
else:
    Name="./Mapping1FaultsInj.sql"


# test_acc = evaluate()
evaluate(net_copy,device,val_loader,mapping,all_connections,IdLookup,neuron_lookup,args.T,int(args.num),Name)
# print(f"Test accuracy: {test_acc:.4f}")




