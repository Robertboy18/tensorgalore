import sys
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig
import torch
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb
from neuralop import H1Loss, LpLoss, get_model
from neuralop.training import setup
from neuralop.utils import get_wandb_api_key, count_model_params
from neuralop.data.transforms.data_processors import DataProcessor

from tensor_galore.profiler_trainer import Trainer
from tensor_galore import AdamW, ProfilerTrainer
from tensor_galore.training_utils import get_scheduler
import scipy.io
from neuralop.models import FNO

def load_burgers_mat(data_path, n_train, n_test, batch_train=32, batch_test=100, 
                 time=1, grid=[0,1]):

    data = scipy.io.loadmat(data_path)

    x_train = torch.tensor(data['a'], dtype=torch.float)[0:n_train,:]
    x_test = torch.tensor(data['a'], dtype=torch.float)[n_train:(n_train + n_test),:]

    y_train = torch.tensor(data['u'], dtype=torch.float)[0:n_train,:]
    y_test = torch.tensor(data['u'], dtype=torch.float)[n_train:(n_train + n_test),:]

    s = x_train.size(-1)

    if grid is not None:
        grid = torch.linspace(grid[0], grid[1], s + 1)[0:-1].view(1,-1)

        grid_train = grid.repeat(n_train, 1)
        grid_test = grid.repeat(n_test, 1)

        x_train = torch.cat((x_train.unsqueeze(1), grid_train.unsqueeze(1)), 1)
        x_test = torch.cat((x_test.unsqueeze(1), grid_test.unsqueeze(1)), 1)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_train, shuffle=False)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_test, shuffle=False)

    test_loaders = {"test": test_loader}
    return train_loader, test_loaders

class BurgersDataProcessor(DataProcessor):
    def __init__(self, device):
        super().__init__()
        self.device = device
    
    def preprocess(self, sample):
        x = sample[0]
        y = sample[1]
        x = x.to(self.device)
        y = y.to(self.device)
        sample = {'x': x, 'y': y}
        sample = {
            k: v.to(self.device)
            for k, v in sample.items()
            if torch.is_tensor(v)
        }

        # may also 
        return sample

    def postprocess(self, out, sample):
        return out, sample

# Read the configuration
config_name = "default"
pipe = ConfigPipeline(
    [
        YamlConfig(
            "./burgers_galore_config.yaml", config_name="default", config_folder="./config"
        ),
        ArgparseConfig(infer_types=True, config_name=None, config_file=None),
        YamlConfig(config_folder="./config"),
    ]
)
config = pipe.read_conf()
config_name = pipe.steps[-1].config_name

logging_name = ""
if config.wandb.name:
    logging_name += config.wandb.name
if config.opt.galore:
    if config.opt.naive_galore:
        logging_name += "_matrixgalore"
    else:
        logging_name += "_tensorgalore"
    if config.opt.act_checkpoint:
        logging_name += "_act_ckpt"


# Set-up distributed communication, if using
device, is_logger = setup(config)

# Set up WandB logging
wandb_args = None
if config.wandb.log and is_logger:
    wandb.login(key=get_wandb_api_key())
    if config.wandb.name:
        wandb_name = logging_name
    else:
        wandb_name = "_".join(
            f"{var}"
            for var in [
                config_name,
                config.fno2d.n_layers,
                config.fno2d.hidden_channels,
                config.fno2d.n_modes_width,
                config.fno2d.n_modes_height,
                config.fno2d.factorization,
                config.fno2d.rank,
                config.patching.levels,
                config.patching.padding,
            ]
        )
    wandb_args =  dict(
        config=config,
        name=wandb_name,
        group=config.wandb.group,
        project=config.wandb.project,
        entity=config.wandb.entity,
    )
    if config.wandb.sweep:
        for key in wandb.config.keys():
            config.params[key] = wandb.config[key]

# Make sure we only print information when needed
config.verbose = config.verbose and is_logger

# Print config to screen
if config.verbose and is_logger:
    pipe.log()
    sys.stdout.flush()

data_path = "/pscratch/sd/r/rgeorge/data/burgers_data_R10.mat"
train_loader, test_loaders = load_burgers_mat(data_path, 800, 200)
output_encoder = None
data_processor = BurgersDataProcessor(device=device)

#torch.cuda.memory._record_memory_history()
model = FNO(
    max_n_modes=(config.fno1d.modes,),
    n_modes=(config.fno1d.modes,),
    hidden_channels=config.fno1d.hidden_channels,
    in_channels=config.fno1d.in_channels,
    out_channels=config.fno1d.out_channels,
)

model = model.to(device)
#torch.cuda.memory._record_memory_history(max_entries=100000)
# Use distributed data parallel
if config.distributed.use_distributed:
    model = DDP(
        model, device_ids=[device.index], output_device=device.index, static_graph=True
    )

if config.opt.galore:
    galore_params = []
    galore_params.extend(list(model.fno_blocks.convs.parameters()))
    print(galore_params[0].shape, galore_params[1].shape, galore_params[2].shape, galore_params[3].shape)
    # drop the first projection layer
    galore_params.pop(0)
    id_galore_params = [id(p) for p in galore_params]
    # make parameters without "rank" to another group
    regular_params = [p for p in model.parameters() if id(p) not in id_galore_params]

    # different tensor vs matrix rank call
    if config.opt.naive_galore:
        galore_rank = config.opt.matrix_galore_rank
    else:
        galore_rank = config.opt.tensor_galore_rank

    if config.opt.per_layer_opt:
        optimizer_dict, scheduler_dict = AdamW.per_layer_weight_opt(model=model,
                                                    id_galore_params=id_galore_params,
                                                    update_proj_gap=config.opt.update_proj_gap,
                                                    galore_scale=config.opt.galore_scale,
                                                    rank=galore_rank,
                                                    weight_decay=config.opt.weight_decay,
                                                    lr=config.opt.lr,
                                                    proj_type=config.opt.galore_proj_type,
                                                    matrix_only=config.opt.naive_galore,
                                                    first_dim_rollup=config.opt.first_dim_rollup,
                                                    scheduler_name=config.opt.scheduler,
                                                    gamma=config.opt.gamma,
                                                    patience=config.opt.scheduler_patience,
                                                    T_max=config.opt.scheduler_T_max,
                                                    step_size=config.opt.step_size
                                                    )
        optimizer = None
        scheduler = None
    else:
        # create a single galore_adamw for all model params
        param_groups = [{'params': regular_params}, 
                        {'params': galore_params, 'type': "tucker", 'rank': galore_rank,\
                        'update_proj_gap': 50, 'scale': config.opt.galore_scale, 'proj_type': "std", 'dim': 5}]
        
        param_groups1 = [{'type': "tucker", 'rank': galore_rank , 'update_proj_gap': 50, \
                        'scale': config.opt.galore_scale, 'proj_type': "std", 'dim': 5}]
        
        
        optimizer = AdamW(param_groups, lr=config.opt.lr, 
                        activation_checkpoint=config.opt.act_checkpoint, 
                        matrix_only=config.opt.naive_galore, 
                        first_dim_rollup=config.opt.first_dim_rollup)
else:
    # Create the optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.opt.learning_rate,
        weight_decay=config.opt.weight_decay,
    )
    param_groups1 = [{'rank': 'baseline'}]

if config.opt.per_layer_opt:
    assert config.opt.scheduler != "ReduceLROnPlateau", "Error: opt/scheduler hooks do not currently support ReduceLROnPlateau."
else:
    scheduler = get_scheduler(
        scheduler_name=config.opt.scheduler,
        optimizer=optimizer,
        gamma=config.opt.gamma,
        patience=config.opt.scheduler_patience,
        T_max=config.opt.scheduler_T_max,
        step_size=config.opt.step_size
    )


# Creating the losses
l2loss = LpLoss(d=2, p=2)
h1loss = H1Loss(d=2)
if config.opt.training_loss == "l2":
    train_loss = l2loss
elif config.opt.training_loss == "h1":
    train_loss = h1loss
else:
    raise ValueError(
        f'Got training_loss={config.opt.training_loss} '
        f'but expected one of ["l2", "h1"]'
    )
eval_losses = {"h1": h1loss, "l2": l2loss}

if config.verbose and is_logger:
    print("\n### MODEL ###\n", model)
    print("\n### OPTIMIZER ###\n", optimizer)
    print("\n### SCHEDULER ###\n", scheduler)
    print("\n### LOSSES ###")
    print(f"\n * Train: {train_loss}")
    print(f"\n * Test: {eval_losses}")
    print(f"\n### Beginning Training...\n")
    sys.stdout.flush()

if config.profile:
    trainer = ProfilerTrainer(
        model=model,
        n_epochs=config.opt.n_epochs,
        device=device,
        data_processor=data_processor,
        amp_autocast=config.opt.amp_autocast,
        wandb_log=config.wandb.log,
        eval_interval=config.wandb.log_test_interval,
        log_output=config.wandb.log_output,
        use_distributed=config.distributed.use_distributed,
        verbose=config.verbose and is_logger,)
else:
    trainer = Trainer(
        model=model,
        n_epochs=config.opt.n_epochs,
        device=device,
        data_processor=data_processor,
        amp_autocast=config.opt.amp_autocast,
        wandb_log=config.wandb.log,
        eval_interval=config.wandb.log_test_interval,
        log_output=config.wandb.log_output,
        use_distributed=config.distributed.use_distributed,
        verbose=config.verbose and is_logger,
    )

# Log parameter count
if is_logger:
    n_params = count_model_params(model)

    if config.verbose:
        print(f"\nn_params: {n_params}")
        sys.stdout.flush()

    if config.wandb.log:
        to_log = {"n_params": n_params}
        if config.n_params_baseline is not None:
            to_log["n_params_baseline"] = (config.n_params_baseline,)
            to_log["compression_ratio"] = (config.n_params_baseline / n_params,)
            to_log["space_savings"] = 1 - (n_params / config.n_params_baseline)
        wandb.log(param_groups1[0])
        wandb.log(to_log)
        wandb.watch(model)

trainer.train(
    train_loader=train_loader,
    test_loaders=test_loaders,
    optimizer=optimizer,
    per_layer_opt=config.opt.per_layer_opt,
    scheduler=scheduler,
    regularizer=False,
    training_loss=train_loss,
    eval_losses=eval_losses,
    burgers=True
)
if config.wandb.log and is_logger:
    wandb.finish()

#torch.cuda.memory._dump_snapshot(f"./{logging_name}_snapshot.html")
