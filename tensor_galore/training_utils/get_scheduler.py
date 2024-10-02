import torch

def get_scheduler(scheduler_name: str,
                  optimizer: torch.optim.Optimizer,
                  gamma: float,
                  patience: int,
                  T_max: int,
                  step_size: int,):
    '''
    Returns LR scheduler of choice from available options
    '''
    if scheduler_name == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=gamma,
            patience=patience,
            mode="min",
        )
    elif scheduler_name == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=T_max
        )
    elif scheduler_name == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )
    else:
        raise ValueError(f"Got scheduler={scheduler_name}")

    return scheduler