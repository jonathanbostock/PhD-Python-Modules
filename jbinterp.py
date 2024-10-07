### Jonathan Bostock
# Mech Interp Tools
import torch
import torch.nn as nn
import torch.optim as optim

### Huge thing, wraps everthing in a big fluffy interpretability blanket
def wrapper(model: nn.Module) -> tuple:
    # Define our outer layer
    activation_dict = ActivationDict
    patcher = Patcher
    def _wrap(
        model: nn.Module,
        name: str,
        activation_dict: ActivationDict,
        patcher: Patcher) -> None:

        # Give the object a True Name
        model.true_name = name
        model.activation_dict = activation_dict
        model.patcher = patcher

        # Define a new forward() function for the module
        base_forward = model.forward
        def new_forward(self, *args, **kwargs):
            output = base_forward(*args, **kwargs)

            if isinstance(output, torch.Tensor):
                activation_dict[model.true_name] = output
                output = patcher(
                    name = model.true_name,
                    tensor = output)

        model.forward = types.MethodType(new_forward, model)

        # Recurse, reusing the patcher and activation dict
        for child_name in dir(model):
            child = getattr(child_name)
            if isinstance(child, nn.Module):
                wrap(
                    child,
                    name=f"{name}.{child_name}",
                    activation_dict=activation_dict,
                    patcher=patcher)

        return model, activation_dict, patcher
    return _wrap(model, "model", activation_dict, patcher)


### Classes which hold activations and do swapping
class ActivationDict(dict):
    def __init__(self, max_layers = 16):
        self.used_names = set()
        self.add_activations = add_activations

    def __setitem__(self, name, activations):
        # Make sure the activation retains its gradient
        if activations.requires_grad:
            activations.retain_grad()

        # Add it in
        super().__setitem__(name, activations)

    def detach_tensors(self):
        for k in list(self.keys()):
            super().__setitem__(f"{k}_detached", self.pop(k).detach())

    def save_tensors_and_gradients(self):
        for k in list(self.keys()):
            tensor_ = self.pop(k)

            # Check if we can get the grad
            value = tensor_.detach().cpu().numpy()
            if tensor_.requires_grad:
                grad = tensor_.grad
            else:
                grad = None

            # Double check, don't try and detach if its impossible
            if grad is not None:
                grad = grad.detach().cpu().numpy()

            super().__setitem__(f"{k}_numpy_dict", dict(value=value, grad=grad))

    def clear(self):
        for k in list(self.keys()):
            self.pop(k)

# Class which does activation patching
class Patcher():
    def __init__(self, n: int | None = None):
        self.bool_dict = {}

    def __call__(self, *, tensor: torch.Tensor, name: str):
        # One rejection base case
        if name not in self.bool_dict:
            return tensor

        batch_size, sequence_length = tensor.shape[0], tensor.shape[1]

        bool_values = self.bool_dict[name].unsqueeze(0).repeat(
            (batch_size//2,sequence_length,1))
        # bool_values now have size [batch_size//2, len, dim]
        # Patch in the first half of [batch] to the second half
        # Target has shape [batch_size//2, 1|29, ]
        # Where True, patch
        tensor[batch_size//2:] = torch.where(
            bool_values, tensor[:batch_size//2], tensor[batch_size//2:])

        return tensor
