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

def batch_to_device(batch,device,non_blocking=False):
    device_batch = {
        "obs": observation_to_device(batch["obs"],device,non_blocking=non_blocking),
        "actions":batch["actions"].to(device, non_blocking=non_blocking),
    }
    if "pad_mask" in batch:
        device_batch["pad_mask"] = batch["pad_mask"].to(device,non_blocking)
    return device_batch

def compute_bc_loss(predicted_actions,target_actions, pad_mask=None):
    """
    Compute action MSE for MLP or RNN policies.

    MLP:

        predicted_actions: [B,A]
        target_actions:    [B,A]
        pad_mask:          None

    RNN:

        predicted_actions: [B,T,A]
        target_actions:    [B,T,A]
        pad_mask:          None or [B,T,1]

    Returns:

        loss:
            Scalar tensor used for backward().

        loss_weight:
            Number of scalar action elements included
            in the loss. Used to calculate an accurate
            epoch average.
    """
    squared_error = F.mse_loss(predicted_actions,target_actions,reduction="none")
    if pad_mask is None:
        loss = squared_error.mean()
        loss_weight = squared_error.numel()
        return loss,loss_weight
    
    if pad_mask.ndim==2:
        pad_mask = pad_mask.unsqueeze(-1)

    pad_mask = pad_mask.to(dtype=squared_error.dtype)
    expanded_mask = pad_mask.expand_as(squared_error)
    loss_weight_tensor = expanded_mask.sum()

    loss = (squared_error*expanded_mask).sum()/loss_weight_tensor.clamp_min(1.0)
    loss_weight = float(loss_weight_tensor.detach())
    return loss, loss_weight

def train_one_epoch(policy, normalizer,data_loader,optimizer,device,max_grad_norm=None):
    policy.train()
    normalizer.eval()

    total_loss =0.0
    total_loss_weight = 0.0
    total_samples = 0
    total_grad_norm = 0.0
    num_updates = 0

    for raw_batch in data_loader:
        batch =batch_to_device(raw_batch,device,True)
        normalized_obs = normalizer(batch["obs"])
        predicted_actions = policy(normalized_obs)
        # loss = F.mse_loss(predicted_actions, batch["actions"])
        loss, loss_weight = compute_bc_loss(predicted_actions,batch["actions"],pad_mask=None)

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

        # batch_size = batch["actions"].shape[0]

        total_loss+=(loss.item()*loss_weight)
        total_loss_weight+=loss_weight

        # total_samples+=batch_size
        num_updates+=1
    
    metrics = {
        "loss": total_loss/total_loss_weight,
        "num_updates":num_updates
    }

    if max_grad_norm is not None:
        metrics["grad_norm"] = (total_grad_norm/max(num_updates,1))
    
    return metrics

@torch.no_grad()
def validate(policy, normalizer,data_loader,device):
    policy.eval()
    normalizer.eval()

    total_loss =0.0
    total_loss_weight = 0
    num_batches=0

    for raw_batch in data_loader:
        batch = batch_to_device(raw_batch,device, True)
        normalized_obs = normalizer(batch["obs"])
        predicted_actions = policy(normalized_obs)
        # loss = F.mse_loss(predicted_actions, batch["actions"])
        loss, loss_weight = compute_bc_loss(predicted_actions,batch["actions"],pad_mask=None)

        # batch_size = batch["actions"].shape[0]

        total_loss += (
            loss.item() * loss_weight
        )

        # total_samples += batch_size
        total_loss_weight+=loss_weight
        num_batches+=1

    return {
        "loss": total_loss / total_loss_weight,
        "num_batches":num_batches
    }

def train_rnn_one_epoch(
    policy,
    normalizer,
    data_loader,
    optimizer,
    device,
    max_grad_norm=1.0,
    use_pad_mask=False
):
    policy.train()
    normalizer.eval()

    total_loss =0.0
    total_loss_weight =0.0
    total_grad_norm = 0.0
    num_updates=0

    for raw_batch in data_loader:
        batch = batch_to_device(raw_batch,device,True)
        normalized_obs = normalizer(batch["obs"])
        predicted_actions,_ = policy(normalized_obs,hidden_state=None)
        pad_mask = None

        if use_pad_mask:
            pad_mask=batch["pad_mask"]
        
        loss,loss_weight = compute_bc_loss(predicted_actions,batch["actions"],pad_mask)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if max_grad_norm is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(),max_grad_norm)
            total_grad_norm+=float(grad_norm)
        
        optimizer.step()
        total_loss+=(loss.item()*loss_weight)
        total_loss_weight+=loss_weight
        num_updates+=1

    metrics = {
        "loss": total_loss/total_loss_weight,
        "num_updates":num_updates
    }

    if max_grad_norm is not None:
        metrics["grad_norm"]= total_grad_norm/max(num_updates,1)
    
    return metrics

@torch.no_grad()
def validate_rnn(policy, normalizer,data_loader,device,use_pad_mask=False):
    policy.eval()
    normalizer.eval()

    total_loss = 0.0
    total_loss_weight=0.0
    num_batches = 0

    for raw_batch in data_loader:
        batch=batch_to_device(raw_batch,device,True)
        normalized_obs = normalizer(batch["obs"])
        predicted_actions,_ = policy(normalized_obs,hidden_state=None)
        pad_mask = None

        if use_pad_mask:
            pad_mask=batch["pad_mask"]
        
        loss,loss_weight = compute_bc_loss(predicted_actions,batch["actions"],pad_mask)

        total_loss+=(loss.item()*loss_weight)
        total_loss_weight+=loss_weight
        num_batches+=1
        
    metrics = {
        "loss": total_loss/total_loss_weight,
        "num_batches":num_batches
    }

    
    return metrics