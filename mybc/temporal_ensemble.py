from collections import deque
import torch

class TemporalActionEnsembler:
    """
    Fuse overlapping action-chunk predictions.

    At environment timestep t, multiple
    previously predicted chunks may contain
    a prediction for action[t]. These
    predictions are combined using
    exponential weights.

    The official ACT convention is:

        weight_i = exp(-coefficient * i)

    where i=0 represents the oldest valid
    prediction.
    """

    def __init__(self,chunk_size,action_dim,coefficient=0.01):
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.coefficient = coefficient
        self._predictions = deque(maxlen=self.chunk_size)

    def reset(self):
        self._predictions.clear()

    def update(self,action_chunk,timestep):
        """
        Add one newly predicted action chunk
        and return the ensembled action for
        the current timestep.

        Args:
            action_chunk:
                [chunk_size, action_dim]

                Preferably normalized actions.

            timestep:
                Current environment timestep.

        Returns:
            action:
                [action_dim]
        """
        if action_chunk.ndim == 3:
            action_chunk=action_chunk.squeeze(0)
        timestep = int(timestep)

        self._predictions.append(
        (
            timestep,
            action_chunk.detach().clone(),
        )
    )
        candidate_actions = []

        for(prediction_timestep,predicted_chunk) in self._predictions:
            chunk_index = timestep-prediction_timestep
            if(0<=chunk_index<self.chunk_size):
                candidate_actions.append(predicted_chunk[chunk_index])

        candidate_actions = torch.stack(candidate_actions,dim=0)
        number_of_candidates = candidate_actions.shape[0]
        prediction_ages = torch.arange(
            number_of_candidates,
            device=candidate_actions.device,
            dtype=candidate_actions.dtype
        )

        weights =torch.exp(-self.coefficient*prediction_ages)
        weights = weights/weights.sum()

        weights = weights.unsqueeze(-1)

        ensembled_action = (candidate_actions*weights).sum(dim=0)
        return ensembled_action
    
    def __len__(self):
        return len(self._predictions)
    