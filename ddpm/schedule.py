import torch

# Global class, just for hinting
class DiffusionSchedule: pass

# Linear Schedule
class LinearSchedule (DiffusionSchedule):
    """
    Linear schedule as in the original DDPM paper.

    Parameters:
    * beta_start : initial value
    * beta_end : final value
    * T : number of diffusion steps
    * device : cpu, gpu or mps
    """

    def __init__(self, beta_start:float, beta_end:float, T:int, device:str):

        # Store the parameters
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.T = T
        self.device = device

        # Compute the schedule
        self.beta = torch.linspace(self.beta_start, self.beta_end, self.T).to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def sample(self, n) -> torch.Tensor:
        "Returns n time step values sampled between 0 and T."
        
        return torch.randint(low=1, high=self.T, size=(n,))