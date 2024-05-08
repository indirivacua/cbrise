import abc
import torch

from stats_running import RunningMeanAndVarianceWelford

def normalize_heatmap(heatmap:torch.Tensor) -> torch.Tensor:
    heatmap_norm = heatmap.clone()
    heatmap_norm -= heatmap_norm.min()
    heatmap_norm /= heatmap_norm.max()
    return heatmap_norm

class StoppingCriteria(abc.ABC):
    @abc.abstractmethod
    def stop(self,iterations:int,heatmap:torch.Tensor):
        pass

class MaxIterations(StoppingCriteria):
    def __init__(self,max_iterations) -> None:
        self.max_iterations=max_iterations
    def stop(self,iterations:int,heatmap:torch.Tensor):
        return iterations>=self.max_iterations

class NoImprovement(MaxIterations):
    def __init__(self,max_iterations:int,patience:int=128,threshold:float=0.1,epsilon:float=1e-3) -> None:
        super().__init__(max_iterations)
        self.patience=patience
        self.epsilon=epsilon
        self.threshold=threshold
        self.heatmap_stats=RunningMeanAndVarianceWelford()
        self.heatmap_stats_prev=None

    def stop(self,iterations:int,heatmap:torch.Tensor):
        convergence = False
        iterations+=1
        if iterations % self.patience == 0:
            heatmap_norm = normalize_heatmap(heatmap)
            self.heatmap_stats.update_batch(heatmap_norm)
            if self.heatmap_stats_prev is not None:
                boolean_tensor = (self.heatmap_stats.var()-self.heatmap_stats_prev).abs() < self.epsilon
                if boolean_tensor.float().mean() > (1 - self.threshold):
                    convergence = True
            self.heatmap_stats_prev=self.heatmap_stats.var()
        return iterations>=self.max_iterations or convergence
