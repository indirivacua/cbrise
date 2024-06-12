import abc
import torch
import torchvision

class Perturbation(abc.ABC):
    @abc.abstractmethod
    def perturb(self,input:torch.Tensor,mask:torch.Tensor):
        pass

class ConstantPerturbation(Perturbation):
    def __init__(self,value) -> None:
        super().__init__()
        self.value=value
    def perturb(self,input:torch.Tensor,masks:torch.Tensor):
        perturbed = torch.zeros(input.shape)*self.value
        return input * masks + perturbed * (1-masks)

class BlurPerturbation(Perturbation):
    def __init__(self,sigmas:torch.Tensor) -> None:
        super().__init__()
        self.sigmas=sigmas
        self.blurred_input = None
    def perturb(self,input:torch.Tensor,masks:torch.Tensor):
        # sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
        kernel_size = (((self.sigmas-0.8)/0.3)+1)*2+1
        kernel_size_int = kernel_size.int()
        kernel_size_int = (kernel_size_int + (kernel_size_int % 2 == 0)) #must be an odd integer
        kernel_size_int = kernel_size_int.tolist()
        perturbed = torchvision.transforms.functional.gaussian_blur(input,kernel_size_int)
        return input * masks + perturbed * (1-masks)
