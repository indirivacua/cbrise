import abc
import torch

class StoppingCriteria(abc.ABC): 
    @abc.abstractmethod()
    def stop(self,iterations:int,heatmap:torch.tensor):
        pass
    
class MaxIterations(StoppingCriteria):
    def __init__(self,max_iterations) -> None:
        self.max_iterations=max_iterations
    def stop(self,iterations:int,heatmap:torch.tensor):
        return iterations<self.max_iterations

class NoImprovement(MaxIterations):
    def __init__(self,max_iterations:int,patience:int,threshold:float) -> None:
        super(max_iterations)
        self.patience=patience
        self.patience_counter=0
        self.threshold=threshold
        self.last=None
        
    def stop(self,iterations:int,heatmap:torch.tensor):
        convergence = False
        if self.last:
            if (self.last-heatmap).abs().mean()<(1-self.threshold):
                self.patience_counter+=1
            else:
                self.patience_counter=0
            convergence = self.patience_counter==self.patience
        self.last=heatmap
        return iterations>=self.max_iterations or convergence
    