### Jonathan Bostock
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Function
Parameter = torch.nn.Parameter

import numpy  as np
from icecream import ic

## This assembles STCs into an interpretable ResNet replacement
class ChRIS(nn.Module):

    def __init__(self, device="cuda"):
        super(ChRIS, self).__init__()
        self.device = device
        self.initialized=False

    def assemble(self, *, base_model, transcoders):

        # This is very oddly named, but it's for _ChRISLayer compatibility
        base_model_modified_dict = {
            "W_dec":    base_model["input_ff.weight"],
            "b_dec":    base_model["input_ff.bias"],
            "W_enc":    base_model["output_ff.weight"],
            "b_enc":    base_model["output_ff.bias"]
        }

        dicts = [base_model_modified_dict, *transcoders]

        # Make a list of layers
        layers = [
            _ChRISLayer(
                state_dict_1=dicts[i],
                state_dict_2=dicts[i+1],
                device=self.device)
            for i in range(len(dicts)-1)]

        # Add the final layer
        layers.append(_ChRISLayer(
            state_dict_1 = dicts[-1],
            state_dict_2 = dicts[0],
            device=self.device,
            nonlinear=False))
        self.layers = nn.ModuleList(layers)

        self.feature_counts = [l.features for l in self.layers]

        self.initialized=True

    def from_state_dict(self, state_dict):

        # Assume last layer is linear, others are not
        # This means we have three parameters per layer
        num_layers = int((len(state_dict)+1)/3)

        layers = []
        for i in range(num_layers):
            layer_name = f"layers.{i}"

            nonlinear = (i != num_layers - 1)

            if nonlinear:
                new_state_dict = {"W":      state_dict[f"{layer_name}.W"],
                                  "b_mag":  state_dict[f"{layer_name}.b_mag"],
                                  "b_gat":  state_dict[f"{layer_name}.b_gat"]}
            else:
                new_state_dict = {"W":      state_dict[f"{layer_name}.W"],
                                  "b":      state_dict[f"{layer_name}.b"]}

            layers.append(_ChRISLayer(
                state_dict_1 = new_state_dict,
                state_dict_2 = None,
                nonlinear = nonlinear,
                device=self.device))

        self.layers = nn.ModuleList(layers)
        self.feature_counts = [l.features for l in self.layers]
        self.initialized=True


    def forward(self, x, return_features = False):
        assert self.initialized, "You need to initialize the model first!"

        features = [x.detach().cpu().numpy()]

        for layer in self.layers:
            x = layer(x)
            if return_features:
                features.append(x.detach().cpu().numpy())

        if return_features:
            return features
        else:
            return x

    def prune(self, epsilon = 1e-2):
        for layer in self.layers:
            W_copy = layer.W.data.clone().detach()
            W_copy[torch.abs(W_copy) < epsilon] = 0
            layer.W.data = W_copy

    def remove_dead(self, feature_activation_counts, epsilon=0):

        feature_activation_list = [None]
        for l in self.layers[:-1]:
            feature_activation_list.append(feature_activation_counts[:l.features])
            feature_activation_counts = feature_activation_counts[l.features:]
        feature_activation_list.append(None)

        for i in range(len(self.layers)):
            l = self.layers[i]
            l.remove_dead(feature_activation_list[i],
                          feature_activation_list[i+1])
            l.features = l.W.size(0)

        self.feature_counts = [l.features for l in self.layers]

    def get_connection_matrices(self):

        return [
            l.W.detach().cpu().numpy() for l in self.layers]

    def parameter_count(self):
        count = 0
        for p in self.parameters():
            count += torch.sum(p != 0).item()
        return count

class _ChRISLayer(nn.Module):
    def __init__(self,*,state_dict_1, state_dict_2, nonlinear=True, device="cuda"):
        super(_ChRISLayer, self).__init__()

        self.nonlinear=nonlinear
        self.linear= not self.nonlinear
        self.device = device

        if state_dict_2 is None:
            self.from_state_dict(state_dict_1, nonlinear, device)
        else:
            self.from_stc_pair(state_dict_1, state_dict_2, nonlinear, device)

        # Define our activation functions
        self.relu = nn.ReLU()
        self.heaviside = lambda x: torch.heaviside(
            x, torch.tensor([0.0], device=self.device)).detach()

    def from_state_dict(self, state_dict, nonlinear=True, device="cuda"):

        self.W = Parameter(state_dict["W"])
        self.features = state_dict["W"].size(0)

        if self.nonlinear:
            self.b_mag = Parameter(state_dict["b_mag"])
            self.b_gat = Parameter(state_dict["b_gat"])
        else:
            self.b = Parameter(state_dict["b"])

    def from_stc_pair(self, state_dict_1, state_dict_2, nonlinear=True, device="cuda"):

        # Big change for linear (final) layers vs nonlinear
        W_dec_1 = state_dict_1["W_dec"]
        b_dec_1 = state_dict_1["b_dec"]
        W_enc_2 = state_dict_2["W_enc"]
        self.features = W_enc_2.size(0)

        if self.nonlinear:
            b_mag_2 = state_dict_2["b_mag"]
            r_mag_2 = state_dict_2["r_mag"]
            b_gat_2 = state_dict_2["b_gat"]

            exp_r = torch.exp(r_mag_2)
            self.W = Parameter((exp_r * (W_enc_2 @ W_dec_1).T).T)
            b_enc = exp_r * (W_enc_2 @ b_dec_1)
            self.b_mag = Parameter(b_enc + b_mag_2)
            self.b_gat = Parameter(b_enc + exp_r * b_gat_2)

            # Define forward
            self.forward = self.forward_nonlinear

        else:
            b_enc_2 = state_dict_2["b_enc"]

            self.W = Parameter(W_enc_2 @ W_dec_1)
            self.b = Parameter(W_enc_2 @ b_dec_1+ b_enc_2)
            self.forward = self.forward_linear

        # Register a hook to maintain sparsity
        # self.W.register_hook(lambda x: self.backward_hook(self, x))

    def forward_nonlinear(self, x):

        x_enc = x @ self.W.T
        via_mag = self.relu(x_enc + self.b_mag)
        via_gat = self.heaviside(x_enc + self.b_gat)

        return via_mag * via_gat

    def forward_linear(self, x):

        x_enc = x @ self.W.T
        return x_enc + self.b

    def forward(self, x):

        if self.linear:
            self.forward = self.forward_linear
        else:
            self.forward = self.forward_nonlinear

        return self.forward(x)

    def remove_dead(self, previous_counts, our_counts,epsilon=0):

        ## Remove dead features
        if previous_counts is not None:
            previous_alive_features = previous_counts > epsilon
            self.W.data = self.W.data[..., previous_alive_features]

        if our_counts is not None:
            alive_features = our_counts > epsilon
            self.W.data = self.W.data[alive_features]
            if self.linear:
                self.b.data = self.b.data[alive_features]
            else:
                self.b_mag.data = self.b_mag.data[alive_features]
                self.b_gat.data = self.b_gat.data[alive_features]

    # I literally ahve no idea how this works
    @staticmethod
    def backward_hook(self, grad):
        grad_copy = grad.clone()
        grad_copy[self.W == 0] = 0
        return grad_copy

### This is to store a bunch of stuff
class BaseSAE(nn.Module):
    """
    Various SAEs inherit from this
    It can function as an SAE (poorly, relatively)
    """
    def __init__(self, *,
                 n_dimensions: int,
                 n_features:int,
                 token_bias: bool = False,
                 vocab_size: int = None,
                 tied_embeddings: bool = False,
                 device: str = "cuda"):
        super(BaseSAE, self).__init__()

        self.n_dimensions = n_dimensions
        self.n_features = n_features

        self.token_bias = token_bias
        self.vocab_size = vocab_size
        self.tied_embeddings = tied_embeddings

        self.device = device
        self.relu = nn.ReLU()

        self.reset_parameters()

        self.r = n_features/n_dimensions

    def reset_parameters(self):

        # Just makes some random parameters
        W_dec_values = torch.randn(
            self.n_dimensions, self.n_features, device=self.device)
        W_dec_norms = torch.linalg.vector_norm(W_dec_values, dim=0)
        W_dec_values = W_dec_values * (1/W_dec_norms)

        self.b_enc = Parameter(
            torch.zeros(self.n_features, device=self.device))


        # Transpose returns a view, so this should work
        # But we have to do it a weird way
        self.W_dec = Parameter(W_dec_values.clone().detach())
        if self.tied_embeddings:
            self.W_enc = W_dec.T
        else:
            self.W_enc = Parameter(W_dec_values.clone().detach().T)


        # Put in an embedding matrix for the tokens
        if self.token_bias:
            self.b_dec = Parameter(torch.zeros(
                (self.vocab_size, self.n_dimensions), device=self.device))
        else:
            self.b_dec = Parameter(
                torch.zeros(self.n_dimensions, device=self.device))

        # Normalize the damn columns
        self.W_dec.register_hook(self.normalize_columns_hook)

    def rescale_parameters(self, scaling_constant):

        W_enc = self.W_enc.data
        b_enc = self.b_enc.data
        W_dec = self.W_dec.data
        b_dec = self.b_dec.data

        W_dec_norms = torch.linalg.vector_norm(W_dec, dim=0)

        # Dimensions of matrices
        # W_enc -> [features, dimension]
        # b_enc -> [(tokens), features]
        # W_dec -> [dimension, features]
        # W_dec_norms -> [features]
        # b_dec -> [dimension]

        self.W_dec.data = (W_dec * (1/(W_dec_norms*scaling_constant))).detach()
        self.b_dec.data = b_dec / scaling_constant
        self.W_enc.data = (W_enc.T * (W_dec_norms*scaling_constant)).T
        self.b_enc.data = b_enc * W_dec_norms - b_dec @ W_enc / scaling_constant

    def forward(self, x):

        if self.token_bias:
            dec_bias = self.b_dec[tokens]
        else:
            dec_bias = self.b_dec

        x_enc = (x - dec_bias) @ self.W_enc.T + self.b_enc
        activations = self.relu(x_enc)
        x_dec = activations @ self.W_dec.T + dec_bias

        if self.training:
            return x_dec
        else:
            return x_dec, activations

    def loss(self, x, y, l1):

        x_enc = x @ self.W_enc.T + self.b_enc
        activations = self.relu(x_enc)

        x_dec = activations @ self.W_dec.T + self.b_dec

        reconstruction_loss = torch.mean((y - x_dec)**2)
        l1_loss = torch.mean(activations) * l1 * self.r

        return reconstruction_loss + l1_loss

    def normalize_columns_hook(self, grad):
        # Normalize the columns of self.W_dec to have unit norm
        W_dec_norm = torch.norm(self.W_dec.data, p=2, dim=0, keepdim=True)
        W_dec_norm[W_dec_norm < 1] = 1
        self.W_dec.data = self.W_dec.data / W_dec_norm

        return grad

# This implements the straight-through estimator
class HeavisideSTEFunction(Function):
    @staticmethod
    def forward(ctx, x, epsilon):
        ctx.save_for_backward(x, epsilon)
        return (x > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        x, epsilon = ctx.saved_tensors
        grad_input = torch.zeros_like(x)
        mask = (x.abs() < epsilon)
        grad_input[mask] = 1.0 / (2 * epsilon)
        return grad_input * grad_output, None


### Jump ReLU SAE Stuff
# Define a Heaviside function
class HeavisideSTE(torch.nn.Module):
    def __init__(self, epsilon):
        super(HeavisideSTE, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        return HeavisideSTEFunction.apply(x, self.epsilon)

## JumpSAE
class JumpSAE(BaseSAE):

    """
    The Jump-ReLU SAE comes from DeepMind (2024)
    https://arxiv.org/abs/2407.14435
    This has the best combination of performance and elegance
    We give options for a per-token decoder bias (improves performance)
    And tied encoder/decoder weights (reduces feature absorption)
    """

    def __init__(self, *,
                 n_dimensions: int,
                 n_features: int,
                 epsilon: float = 1e-2,
                 token_bias: bool = False,
                 vocab_size: int = None,
                 tied_embeddings: bool = False,
                 device: str = "cuda"):
        super(JumpSAE, self).__init__(
            n_dimensions = n_dimensions,
            n_features = n_features,
            token_bias = token_bias,
            vocab_size = vocab_size,
            tied_embeddings = tied_embeddings,
            device = device)

        self.heaviside_ste = HeavisideSTE(epsilon=torch.tensor(epsilon, device=device))
        self.one = torch.tensor(1, device=self.device)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Resets the parameters found in BaseSAE
        Then resets theta to also be zero
        """
        super().reset_parameters()

        self.theta = Parameter(
            torch.zeros(self.n_features, device=self.device))

    def rescale_parameters(self) -> None:
        """
        We use a backward hook so we don't actually need this
        This function just passes
        """
        pass

    def forward(self, x: torch.Tensor, tokens: torch.Tensor=None):
        """
        Algorithm is:
            subtract decoder bias (possibly per-token)
            project up into feature space
            add encoder bias
            apply ReLU
            gate by theta
            project down to input space
            add decoder bias again
        """
        if self.token_bias:
            dec_bias = self.b_dec[tokens]
        else:
            dec_bias = self.b_dec

        preactivations = (x - dec_bias) @ self.W_enc.T + self.b_enc
        gate_values = self.heaviside_ste(preactivations - self.theta)
        activations = self.relu(preactivations) * gate_values.detach()
        decoded_values = activations @ self.W_dec.T + dec_bias

        # Return the gate values so we can do our regularization
        return activations, decoded_values, gate_values

    def loss(self, *, x: torch.Tensor, y: torch.Tensor,
             target_l0: float, tokens: torch.Tensor=None):
        """
        Algorithm is:
            first get our output_estimate (y_estimate)
            penalize by MSE divided by the variance in y (to normalize)
            penalize l0 against the target_l0
            Return both mse and l0 loss
        """
        _, y_estimate, gate_values = self.forward(x, tokens=tokens)

        y_mean = torch.mean(y, dim=0, keepdim=True)
        y_var = (y-y_mean)**2

        mse_loss = torch.mean((y_estimate - y) **2)
        mse_loss_scaled = mse_loss / torch.mean(y_var).detach()

        # Take an l1/l2 interpolated loss on our l0 loss compared to desired l0
        l0 = torch.mean(torch.sum(gate_values, dim=-1))
        l0_loss = (l0/target_l0 - 1)**2

        return mse_loss_scaled, l0_loss
### End JumpSAE


class TopKSAE(BaseSAE):
    # This is taken from OpenAI (2024)
    # https://arxiv.org/html/2406.04093v1
    def __init__(self,*, n_dimensions, n_features, k=8, device="cuda"):
        super(TopKSAE, self).__init__(n_dimensions = n_dimensions,
                                      n_features = n_features,
                                      device=device)

        self.reset_parameters()
        self.k = k

    def reset_parameters(self):

        # Just makes some random parameters
        W_dec_values = torch.randn(
            self.n_dimensions, self.n_features, device=self.device)
        W_dec_norms = torch.linalg.vector_norm(W_dec_values, dim=0)
        W_dec_values = W_dec_values * (1/W_dec_norms)

        self.b_enc = Parameter(
            torch.zeros(self.n_features, device=self.device))
        self.W_enc = Parameter(W_dec_values.clone().detach().transpose(-2,-1))

        self.b_dec = Parameter(
            torch.zeros(self.n_dimensions, device=self.device))
        self.W_dec = Parameter(W_dec_values.clone().detach())

    def rescale_parameters(self, scaling_constant):
        # Exists but left blank for compatibility
        pass

    def forward(self, x):

        ## Have to do x @ W.T because of batching, which is mildly annoying
        enc_x = (x - self.b_dec) @ self.W_enc.T + self.b_enc

        features = self.get_top_k(enc_x)
        predicted_residual = features @ self.W_dec.T + self.b_dec

        if self.training:
            return predicted_residual
        else:
            return predicted_residual, features

    def get_top_k(self, x):
        _, indices = torch.topk(x, k=self.k, dim=1)
        mask = torch.zeros_like(x, dtype=torch.bool)
        mask.scatter_(1, indices, True)
        result = torch.zeros_like(x)
        result[mask] = x[mask]

        return result

    def loss(self, x, y, k):

        # Kinda hacky to make it compatible with existing stuff
        self.k = int(k)

        ### All losses are calculated as their means over the relevant dimensions
        # Shouldn't cause issues since we aren't regularizing
        enc_x = (x - self.b_dec) @ self.W_enc.T + self.b_enc

        activations = self.get_top_k(enc_x)
        reconstruction = activations @ self.W_dec.T + self.b_dec

        loss = torch.mean((y - reconstruction)**2)

        # Auxiliary loss to prevent feature death, uses the next k features
        # Kind of weird calcuation, this has the effect of setting the top k
        # features in enc_x to zero and passing them as aux_x
        aux_x = enc_x - activations

        aux_activations = self.get_top_k(aux_x)
        aux_reconstruction = aux_activations @ self.W_dec.T + self.b_dec

        aux_loss = torch.mean((y - aux_reconstruction)**2)

        return loss + aux_loss
### End TopKSAE ###


### Can also be used as an STC
class GatedSAE(BaseSAE):
    # This is taken from DeepMind (2024)
    # arxiv.org/pdf/2404.16014
    # Initialization is taken from Anthropic (2024)
    # anthropic.com/research/circuits-updates-april-2024
    # Norms of W_dec are clipped to be <= 1 at all times
    def __init__(self,*, n_dimensions, n_features, device="cuda"):
        super(GatedSAE, self).__init__(n_dimensions = n_dimensions,
                                       n_features = n_features,
                                       device=device)
        self.n_dimensions = n_dimensions
        self.n_features = n_features
        self.device=device
        self.reset_parameters()
        self.r = n_features/n_dimensions

        self.resample_max = 10000
        self.resample_counts = np.array([self.resample_max] * n_features)

    def get_new_W_dec(self):

        W_dec_values = torch.randn(
            self.n_dimensions, self.n_features, device=self.device)
        W_dec_norms = torch.linalg.vector_norm(W_dec_values, dim=0)
        W_dec_values = W_dec_values * (0.1/W_dec_norms)

        return W_dec_values

    def reset_parameters(self):

        W_dec_values = self.get_new_W_dec()

        self.b_gat = Parameter(
            torch.zeros(self.n_features, device=self.device))

        self.r_mag = Parameter(
            torch.zeros(self.n_features, device=self.device))

        # self.r_mag = torch.zeros(self.n_features, device=self.device)
        self.b_mag = Parameter(
            torch.zeros(self.n_features, device=self.device))

        self.W_enc = Parameter(W_dec_values.clone().detach().transpose(-2,-1))

        self.b_dec = Parameter(
            torch.zeros(self.n_dimensions, device=self.device))
        self.W_dec = Parameter(W_dec_values.clone().detach())

        self.relu = nn.ReLU()

        ### NB!!! Heaviside kills the gradient (unavoidable)
        self.heaviside = lambda x: torch.heaviside(x, torch.tensor([0.0],
                                                                   device=self.device)).detach()

    # Currently deprecated
    """
    def softexp(self, x, scale=1):

        output = torch.zeros_like(x)

        x_adj = x + self.cutoff
        x_big = (x_adj > 0)

        output[x_big] = x_adj[x_big] + 1
        output[~x_big] = torch.exp(x_adj[~x_big])

        return output * scale * np.exp(self.cutoff)
    """

    def rescale_parameters(self, scaling_constant):

        W_dec_values = self.W_dec.data
        W_dec_norms = torch.linalg.vector_norm(W_dec_values, dim=0)

        # Dimentsions of matrices
        # W_enc -> [features, dimension]
        # W_dec -> [dimension, features]
        # W_dec_norms -> [features]

        self.W_dec.data = (W_dec_values * (1/(W_dec_norms*scaling_constant))).detach()
        self.b_dec.data = self.b_dec.data / scaling_constant
        self.W_enc.data = (self.W_enc.data.T * (W_dec_norms*scaling_constant)).T
        self.b_gat.data = (self.b_gat.data * W_dec_norms).detach()
        self.b_mag.data = (self.b_mag.data * W_dec_norms).detach()

    def forward(self, x):

        ## Have to do x @ W.T because of batching, which is mildly annoying

        enc_x = x @ self.W_enc.T
        mag_path = self.relu(torch.exp(self.r_mag) * enc_x + self.b_mag)
        gat_path = self.heaviside(enc_x + self.b_gat)

        features = mag_path * gat_path
        predicted_residual = features @ self.W_dec.T + self.b_dec

        if self.training:
            return predicted_residual
        else:
            return predicted_residual, features

    def loss(self, x, y, l1, track_resampling = False):

        ### All losses are calculated as their means over the relevant dimensions
        # This reduces the reconstruction losses by n_dimensions
        # but reduces the L1 loss by n_features
        # Therefore we multiply the L1 loss by self.r

        # We have to repeat code from forward but there's no easy way around this
        # First calculate the encoded x and the relevant pre-activations

        enc_x = x @ self.W_enc.T
        gat_preactivations = enc_x + self.b_gat
        mag_preactivations = torch.exp(self.r_mag) * enc_x + self.b_mag

        reconstruction = (self.relu(mag_preactivations) * \
                          self.heaviside(gat_preactivations)) @ self.W_dec.T + self.b_dec

        reconstruction_loss = torch.mean((reconstruction - y)**2)

        # We are constraining W_dec
        # Have to include r in this to adjust for different feature counts
        gate_feature_magnitude_loss = l1 * torch.mean(self.relu(gat_preactivations)) * self.r

        # Ensure that the gate_path has approximately the right magnitude
        gate_reconstruction =  self.relu(gat_preactivations) @ self.W_dec.detach().T + self.b_dec.detach()
        gate_reconstruction_loss = torch.mean((gate_reconstruction - y)**2)

        self.track_resampling(gat_preactivations)

        return reconstruction_loss + gate_feature_magnitude_loss + gate_reconstruction_loss

    def track_resampling(self, features):

        batch_size = features.size(0)
        alive_features = torch.sum(features, 0).detach().cpu().numpy() > 0

        self.resample_counts -= batch_size
        self.resample_counts[self.resample_counts < 0] = 0
        self.resample_counts[alive_features] = self.resample_max

    def resample(self):

        dead_features = self.resample_counts == 0
        if np.sum(dead_features) > self.n_features / 4:
            new_W_dec = self.get_new_W_dec()
            self.W_dec.data[::,dead_features] = new_W_dec[::,dead_features]
            self.W_enc.data[dead_features] = new_W_dec.T[dead_features]
            self.b_gat.data[dead_features] = 0
            self.b_mag.data[dead_features] = 0
            self.r_mag.data[dead_features] = 0
