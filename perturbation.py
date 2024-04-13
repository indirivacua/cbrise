import abc
import torch
import torchvision
class Perturbation(abc.ABC):
    @abc.abstractmethod
    def perturb(self,input:torch.tensor,mask:torch.tensor):
        pass

class ConstantPerturbation(Perturbation):
    def __init__(self,value) -> None:
        super().__init__()
        self.value=value
    def perturb(self,input:torch.tensor,masks:torch.tensor):
        perturbed = torch.zeros(input.shape)*self.value
        return input * masks + perturbed * (1-masks)

class BlurPerturbation(Perturbation):
    def __init__(self,sigmas:torch.tensor) -> None:
        super().__init__()
        self.sigmas=sigmas
        self.blurred_input = None
    def perturb(self,input:torch.tensor,masks:torch.tensor):
        # sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
        kernel_size = (((self.sigmas-0.8)/0.3)+1)*2+1
        kernel_size_int = kernel_size #todo
        perturbed = torchvision.transforms.functional.gaussian_blur(input,kernel_size_int)
        return input * masks + perturbed * (1-masks)
