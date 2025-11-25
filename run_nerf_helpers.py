import torch
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# Misc
img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)


# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


# Learnable Fourier Feature Encoding
class LearnableFourierEmbedder(nn.Module):
    def __init__(self, input_dims=3, num_freqs=10, include_input=True, 
                 learnable_freqs=True, learnable_phase=False, init_scale=1.0, use_gating=False):
        """
        Learnable Fourier Feature Encoding for NeRF
        
        Args:
            input_dims: Input dimension (3 for xyz coordinates)
            num_freqs: Number of frequency bands
            include_input: Whether to include raw input in output
            learnable_freqs: Make frequency bands learnable
            learnable_phase: Make phase shifts learnable
            init_scale: Initial scale for frequency initialization
            use_gating: If True, use learnable per-frequency amplitude gates
        """
        super(LearnableFourierEmbedder, self).__init__()
        self.input_dims = input_dims
        self.num_freqs = num_freqs
        self.include_input = include_input
        self.learnable_freqs = learnable_freqs
        self.learnable_phase = learnable_phase
        self.use_gating = use_gating
        
        # Initialize frequency bands (log-spaced like original NeRF)
        freq_bands = 2.**torch.linspace(0., num_freqs-1, steps=num_freqs) * init_scale
        
        if learnable_freqs:
            # Store frequencies in LOG SPACE for proportional learning
            # This ensures all frequencies update proportionally regardless of magnitude
            # log_freq_bands will be the learnable parameter
            log_freq_bands = torch.log(freq_bands + 1e-8)  # Add small epsilon for numerical stability
            self.log_freq_bands = nn.Parameter(log_freq_bands)
            # freq_bands is computed from log_freq_bands in forward pass
            self.register_buffer('_freq_scale', torch.ones(1))  # For compatibility
        else:
            # Keep frequencies fixed
            self.register_buffer('freq_bands', freq_bands)
            self.log_freq_bands = None
        
        # Initialize phase shifts to zero
        if learnable_phase:
            # Separate phase for sin and cos, for each frequency and dimension
            self.phase_shifts = nn.Parameter(torch.zeros(num_freqs, input_dims, 2))
        else:
            self.register_buffer('phase_shifts', torch.zeros(num_freqs, input_dims, 2))
        
        # Per-frequency amplitude gates (learnable)
        if use_gating:
            # Initialize: low-freq gates ~1, high-freq gates ~0.1 (soft progressive encoding)
            amp_gates_init = torch.ones(num_freqs)
            # Make high-frequency gates smaller initially
            for i in range(num_freqs):
                if i > num_freqs // 2:
                    amp_gates_init[i] = 0.1
            # Store in log space for positive amplitudes
            self.amp_gates = nn.Parameter(torch.log(amp_gates_init + 1e-8))
        else:
            self.amp_gates = None
        
        # Calculate output dimension
        out_dim = 0
        if include_input:
            out_dim += input_dims
        out_dim += num_freqs * input_dims * 2  # *2 for sin and cos
        self.out_dim = out_dim
        
    def forward(self, inputs):
        """
        Args:
            inputs: [..., input_dims] input coordinates
        Returns:
            [..., out_dim] encoded features
        """
        outputs = []
        
        if self.include_input:
            outputs.append(inputs)
        
        # Get frequency bands (from log space if learnable)
        if self.learnable_freqs and hasattr(self, 'log_freq_bands'):
            # Convert from log space to linear space for proportional updates
            freq_bands = torch.exp(self.log_freq_bands) - 1e-8  # Inverse of log
        else:
            freq_bands = self.freq_bands
        
        # Apply learnable Fourier features
        for i, freq in enumerate(freq_bands):
            # Compute frequency-scaled inputs: [..., input_dims]
            scaled_inputs = inputs * freq
            
            # Get amplitude gate for this frequency
            if self.use_gating and self.amp_gates is not None:
                gate = torch.exp(self.amp_gates[i])
            else:
                gate = 1.0
            
            if self.learnable_phase:
                # Add learnable phase shifts
                sin_phase = self.phase_shifts[i, :, 0]  # [input_dims]
                cos_phase = self.phase_shifts[i, :, 1]  # [input_dims]
                outputs.append(gate * torch.sin(scaled_inputs + sin_phase))
                outputs.append(gate * torch.cos(scaled_inputs + cos_phase))
            else:
                outputs.append(gate * torch.sin(scaled_inputs))
                outputs.append(gate * torch.cos(scaled_inputs))
        
        return torch.cat(outputs, -1)


def get_embedder(multires, i=0, learnable=False, learnable_phase=False, 
                 learnable_freqs=True, init_scale=1.0, use_gating=False):
    """
    Get embedder function for positional encoding
    
    Args:
        multires: Number of frequency bands (log2 of max frequency)
        i: Embedding type (0: default, -1: none)
        learnable: Use learnable Fourier features
        learnable_phase: Make phase shifts learnable (only if learnable=True)
        learnable_freqs: Make frequency bands learnable (only if learnable=True)
        init_scale: Initial scale for frequency initialization (only if learnable=True)
        use_gating: Use per-frequency amplitude gates (only if learnable=True)
    
    Returns:
        embed: Embedding function or module
        out_dim: Output dimension of embedding
    """
    if i == -1:
        return nn.Identity(), 3
    
    if learnable:
        # Use learnable Fourier feature encoding
        embedder_obj = LearnableFourierEmbedder(
            input_dims=3,
            num_freqs=multires,
            include_input=True,
            learnable_freqs=learnable_freqs,
            learnable_phase=learnable_phase,
            init_scale=init_scale,
            use_gating=use_gating
        )
        return embedder_obj, embedder_obj.out_dim
    else:
        # Use original fixed positional encoding
        embed_kwargs = {
                    'include_input' : True,
                    'input_dims' : 3,
                    'max_freq_log2' : multires-1,
                    'num_freqs' : multires,
                    'log_sampling' : True,
                    'periodic_fns' : [torch.sin, torch.cos],
        }
        
        embedder_obj = Embedder(**embed_kwargs)
        embed = lambda x, eo=embedder_obj : eo.embed(x)
        return embed, embedder_obj.out_dim


# Model
class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False, use_film=False):
        """ 
        Args:
            use_film: If True, use FiLM-style conditioning instead of concatenation
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        self.use_film = use_film
        
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        
        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            
            if use_film:
                # FiLM-style conditioning: direction MLP generates gamma and beta
                self.dir_mlp = nn.Sequential(
                    nn.Linear(input_ch_views, W),
                    nn.ReLU(inplace=True),
                    nn.Linear(W, 2 * W),  # -> gamma and beta
                )
                self.color_mlp = nn.Sequential(
                    nn.Linear(W, W),
                    nn.ReLU(inplace=True),
                    nn.Linear(W, 3),
                )
            else:
                # Deeper view MLP (paper-style): 2 layers instead of 1
                self.views_linears = nn.ModuleList([
                    nn.Linear(input_ch_views + W, W // 2),
                    nn.Linear(W // 2, W // 2),
                ])
                self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            sigma = self.alpha_linear(h)
            feat = self.feature_linear(h)  # [.., W]
            
            if self.use_film:
                # FiLM modulation by view dir
                gamma_beta = self.dir_mlp(input_views)  # [.., 2W]
                gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)
                feat = gamma * feat + beta
                rgb = self.color_mlp(feat)
            else:
                # Deeper view MLP path
                h = torch.cat([feat, input_views], -1)
                for l in self.views_linears:
                    h = F.relu(l(h))
                rgb = self.rgb_linear(h)
            
            outputs = torch.cat([rgb, sigma], -1)
        else:
            outputs = self.output_linear(h)

        return outputs    

    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"
        
        # Load pts_linears
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linears]))    
            self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears+1]))
        
        # Load feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear+1]))

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears+1]))

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear+1]))

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear+1]))



# Ray helpers
def get_rays(H, W, K, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d
    
    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]
    
    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)
    
    return rays_o, rays_d


# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples
