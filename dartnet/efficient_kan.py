import torch
from torch.nn import functional as F
import math
"""
1. Memory efficiency: The original implementation expands all intermediate variables to apply
  different activations; this code reformulates the computation to activate inputs with different
  basis functions and then linearly combine them. This reformulation can significantly reduce
  memory cost and make computation more efficient.
2. Regularization change: L1 regularization in the original implementation requires nonlinear
  ops on tensors and is incompatible with the reformulated computation. Therefore this code
  switches to L1 regularization on weights, which is more common in neural nets and compatible
  with the reformulation.
3. Activation scaling option: The original implementation includes a learnable scale per
  activation; this library provides an option to disable it. Disabling scaling can make the
  model more efficient but may affect results.
4. Parameter initialization change: To address performance issues on MNIST, this code changes
  parameter initialization to use Kaiming initialization.
"""
 
class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,  # Grid size, default 5
        spline_order=3, # Piecewise polynomial (spline) order, default 3
        scale_noise=0.1,  # Noise scale, default 0.1
        scale_base=1.0,   # Base scale, default 1.0
        scale_spline=1.0,    # Spline scale, default 1.0
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,  # Base activation, default SiLU (Sigmoid Linear Unit)
        grid_eps=0.02,
        grid_range=[-1, 1],  # Grid range, default [-1, 1]
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size # Set grid size and spline order
        self.spline_order = spline_order
 
        h = (grid_range[1] - grid_range[0]) / grid_size   # Grid step size
        grid = ( # Build grid
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)  # Register grid as a buffer
 
        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features)) # Init base and spline weights
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:  # If standalone spline scaling is enabled, init scaler params
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )
 
        self.scale_noise = scale_noise # Store noise/base/spline scales, standalone flag, base activation, and grid eps
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps
 
        self.reset_parameters()  # Reset parameters
 
    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)# Kaiming uniform init for base weights
        with torch.no_grad():
            noise = (# Generate scaled noise
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_( # Compute spline weights
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:  # If standalone spline scaling enabled, Kaiming-uniform init spline scaler
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)
 
    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
 
        grid: torch.Tensor = ( # Shape (in_features, grid_size + 2 * spline_order + 1)
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )
 
        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()
 
    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)
        
        A = self.b_splines(x).transpose(0, 1)  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)

   
        # ========== Added: tiny noise on A to force full rank ==========
        # torch.manual_seed(42)
        noise = torch.randn_like(A) * 1e-6  # 1e-6 Gaussian noise; adjust if needed
        A = A + noise
        # ====================================================
        
        # # ========== Fixed noise: temp seed + restore (compatible with older PyTorch) ==========
        # with torch.no_grad():  # Noise needs no grad; more efficient
        #     # 1. Save current global RNG seed (avoid affecting other code)
        #     original_seed = torch.seed()
            
        #     # 2. Set fixed seed (must match training script global seed, e.g. 42)
        #     # Note: if training uses torch.manual_seed(123), change this to 123 as well
        #     torch.manual_seed(42)
            
        #     # 3. Generate fixed noise (seed fixed => identical noise each run)
        #     noise = torch.randn_like(A) * 1e-6
            
        #     # 4. Restore original global seed (do not break other randomness)
        #     torch.manual_seed(original_seed)
            
        #     # 5. Add noise to A (force full rank) + clamp extremes (avoid numeric blow-up)
        #     A = A + noise
        #     A = torch.clamp(A, min=-1e4, max=1e4)  # Optional but recommended
        # # ====================================================

        solution = torch.linalg.lstsq(A, B).solution  # Solve least squares
        result = solution.permute(2, 0, 1)
        
        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    # def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
    #     """
    #     Compute the coefficients of the curve that interpolates the given points.
    #     Args:
    #         x (torch.Tensor): Input tensor of shape (batch_size, in_features).
    #         y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).
    #     Returns:
    #         torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
    #     """
    #     assert x.dim() == 2 and x.size(1) == self.in_features
    #     assert y.size() == (x.size(0), self.in_features, self.out_features)
    #     # Compute B-spline basis functions
    #     A = self.b_splines(x).transpose(
    #         0, 1 # Shape (in_features, batch_size, grid_size + spline_order)
    #     )  # (in_features, batch_size, grid_size + spline_order)
    #     B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
    #     solution = torch.linalg.lstsq(   # Solve linear system via least squares
    #         A, B
    #     ).solution  # (in_features, grid_size + spline_order, out_features)
    #     result = solution.permute( # Reorder result dimensions
    #         2, 0, 1
    #     )  # (out_features, in_features, grid_size + spline_order)
 
    #     assert result.size() == (
    #         self.out_features,
    #         self.in_features,
    #         self.grid_size + self.spline_order,
    #     )
    #     return result.contiguous()

    
    @property
    def scaled_spline_weight(self):
        """
        Get scaled spline weights.
        Returns:
        torch.Tensor: Scaled spline weight tensor, same shape as self.spline_weight.
        """
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )
 
    def forward(self, x: torch.Tensor): # Pass input through linear transforms and activations to produce output
        """
        Forward pass.
        Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
        torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
 
        base_output = F.linear(self.base_activation(x), self.base_weight) # Base linear branch output
        spline_output = F.linear( # Spline linear branch output
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        return base_output + spline_output  # Sum of base and spline branch outputs
 
    @torch.no_grad()
    # Update the grid.
    # Args:
    # x (torch.Tensor): Input tensor of shape (batch_size, in_features).
    # margin (float): Margin size at grid edges. Default 0.01.
    # Dynamically update the model grid based on the distribution of input x so the model
    # better fits the data distribution and improves expressiveness / generalization.
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)
 
        splines = self.b_splines(x)  # (batch, in, coeff)  # Compute B-spline bases
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)  # Reorder to (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)  # Reorder to (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)
 
        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0] # Sort each channel to collect data distribution
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]
 
        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )
 
        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )
 
        self.grid.copy_(grid.T)   # Update grid and spline weights
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))
 
    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        # Compute regularization loss to constrain parameters and reduce overfitting
        """
        Compute the regularization loss.
        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.
        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )
 
 
class KAN(torch.nn.Module): # Wrapper for a KAN network usable for fitting and prediction.
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        """
        Initialize the KAN model.
        Args:
            layers_hidden (list): List of input feature sizes for each hidden layer.
            grid_size (int): Grid size, default 5.
            spline_order (int): Spline order, default 3.
            scale_noise (float): Noise scale, default 0.1.
            scale_base (float): Base scale, default 1.0.
            scale_spline (float): Spline scale, default 1.0.
            base_activation (torch.nn.Module): Base activation, default SiLU.
            grid_eps (float): Grid blending parameter, default 0.02.
            grid_range (list): Grid range, default [-1, 1].
        """
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order
 
        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )
 
    def forward(self, x: torch.Tensor, update_grid=False): # Call each KANLinear.forward for the forward pass.
        """
        Forward pass.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            update_grid (bool): Whether to update the grid. Default False.
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x
 
    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):# Regularization loss to constrain parameters and reduce overfitting.
        """
        Compute regularization loss.
        Args:
            regularize_activation (float): Weight for activation regularization, default 1.0.
            regularize_entropy (float): Weight for entropy regularization, default 1.0.
        Returns:
            torch.Tensor: Regularization loss.
        """
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )

# class MLP_KAN_Hybrid(nn.Module):
#     def __init__(self, in_dim, hidden_dim, out_dim):
#         super().__init__()
#         self.mlp_linear = nn.Linear(in_dim, hidden_dim)
#         self.mlp_norm = nn.BatchNorm1d(hidden_dim)
#         self.mlp_act = nn.Mish()
#         self.kan = KAN(
#             layers_hidden=[hidden_dim, out_dim],
#             grid_size=5,
#             spline_order=3,
#             base_activation=nn.Mish
#         )
#     def forward(self, x, update_grid=False):
#         x = self.mlp_act(self.mlp_norm(self.mlp_linear(x)))
#         x = self.kan(x, update_grid=update_grid)
#         return x
#     def regularization_loss(self):
#         return self.kan.regularization_loss(regularize_activation=1.0, regularize_entropy=0.1)
