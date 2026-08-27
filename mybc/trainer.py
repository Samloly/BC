from collections import defaultdict
import torch
import torch.nn.functional as F

def observation_to_device(
    observation,
    device,
    non_blocking=False,
):
    return {
        key: value.to(
            device,
            non_blocking=non_blocking
        )
        for key,value in observation.items()
    }

def bath_to_device(batch,device,non_blocking=False):
    return{
        "obs": observation_to_device(batch["obs"],device,non_blocking=non_blocking),
        "actions":batch["actions"].to(device, non_blocking=non_blocking),
    }

def train_one_epoch(policy, normalizer,data_loader,optimizer,device,max_grad_norm=None):
    policy.train()
    normalizer.eval()

    total_loss =0.0
    total_samples = 0
    total_grad_norm = 0.0
    num_updates = 0

    for raw_batch in data_loader:
        batch =bath_to_device(raw_batch,device,True)
        normalized_obs = normalizer(batch["obs"])
        predicted_actions = policy(normalized_obs)
        loss = F.mse_loss(predicted_actions, batch["actions"])

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if max_grad_norm is not None:
            grad_norm = (
                torch.nn.utils.clip_grad_norm_(
                    policy.parameters(),
                    max_grad_norm
                )
            )

            total_grad_norm +=float(grad_norm)

        optimizer.step()

        batch_size = batch["actions"].shape[0]

        total_loss+=(loss.item()*batch_size)

        total_samples+=batch_size
        num_updates+=1
    
    metrics = {"loss": total_loss/total_samples}

    if max_grad_norm is not None:
        metrics["grad_norm"] = (total_grad_norm/max(num_updates,1))
    
    return metrics

@torch.no_grad()
def validate(policy, normalizer,data_loader,device):
    policy.eval()
    normalizer.eval()

    total_loss =0.0
    total_samples = 0

    for raw_batch in data_loader:
        batch = bath_to_device(raw_batch,device, True)
        normalized_obs = normalizer(batch["obs"])
        predicted_actions = policy(normalized_obs)
        loss = F.mse_loss(predicted_actions, batch["actions"])

        batch_size = batch["actions"].shape[0]

        total_loss += (
            loss.item() * batch_size
        )

        total_samples += batch_size

    return {
        "loss": total_loss / total_samples,
    }