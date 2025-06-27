"""
Underwater Image Enhancement using Improved SeaThru Algorithm

This implementation provides a comprehensive framework for underwater image restoration
based on physical models of light propagation in water. The algorithm addresses
backscatter removal, illumination estimation, and attenuation correction.

References:
    - Akkaynak, D., & Treibitz, T. (2019). Sea-thru: A method for removing water from underwater images.
    - Schechner, Y. Y., & Karpel, N. (2005). Recovery of underwater visibility and structure by polarization analysis.

Author: [Your Name]
Date: 2024
"""

# ================================================================================
# Critical: Handle OpenMP conflicts before any scientific library imports
# ================================================================================
import os
import platform

# Resolve OpenMP conflicts on Windows
if platform.system() == 'Windows':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Set optimal thread counts
if 'OMP_NUM_THREADS' not in os.environ:
    import multiprocessing
    num_cores = multiprocessing.cpu_count()
    optimal_threads = str(max(1, min(num_cores - 2, 8)))  # Cap at 8 threads
    os.environ['OMP_NUM_THREADS'] = optimal_threads
    os.environ['MKL_NUM_THREADS'] = optimal_threads
    os.environ['NUMEXPR_NUM_THREADS'] = optimal_threads

# ================================================================================
# Standard imports in specific order to minimize conflicts
# ================================================================================
import numpy as np  # Import NumPy first (often includes MKL)
import cv2
import scipy as sp
import scipy.optimize as opt
import scipy.stats as stats
from scipy.optimize import differential_evolution, curve_fit

# Import scikit-image components
from skimage import exposure, restoration
from skimage.restoration import denoise_bilateral, denoise_tv_chambolle, denoise_wavelet
from skimage.morphology import closing, opening, erosion, dilation, disk, diamond, square
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# Other imports
import collections
from typing import Tuple, Dict, Callable, Optional
import argparse
import sys
import matplotlib.pyplot as plt
from PIL import Image
import warnings
warnings.filterwarnings('ignore')


# ================================================================================
# Physical Models for Underwater Light Propagation
# ================================================================================

class BackscatterModel:
    """
    Implementation of the improved backscatter model for underwater images.
    
    The backscatter component is modeled as:
    B(z) = B_inf * (1 - exp(-beta_B * z^gamma))
    
    where:
        - B_inf: asymptotic backscatter value
        - beta_B: backscatter coefficient
        - gamma: non-linear depth dependency factor
        - z: depth
    """
    
    @staticmethod
    def compute(depth: np.ndarray, B_inf: float, beta_B: float, gamma: float) -> np.ndarray:
        """
        Compute backscatter values for given depths.
        
        Args:
            depth: Depth map (2D array)
            B_inf: Asymptotic backscatter value
            beta_B: Backscatter coefficient
            gamma: Non-linear depth dependency factor
            
        Returns:
            Backscatter values
        """
        return B_inf * (1 - np.exp(-beta_B * depth**gamma))
    
    @staticmethod
    def find_estimation_points(img_channel: np.ndarray, depths: np.ndarray, 
                             num_bins: int = 10, fraction: float = 0.01, 
                             max_vals: int = 20) -> np.ndarray:
        """
        Find points for backscatter estimation using dark channel prior.
        
        Args:
            img_channel: Single channel image
            depths: Depth map
            num_bins: Number of depth bins
            fraction: Fraction of darkest pixels to select
            max_vals: Maximum values per bin
            
        Returns:
            Array of (depth, intensity) pairs
        """
        z_max, z_min = np.max(depths), np.min(depths)
        z_ranges = np.linspace(z_min, z_max, num_bins + 1)
        
        points = []
        for i in range(len(z_ranges) - 1):
            # Select pixels in current depth range
            mask = (depths >= z_ranges[i]) & (depths <= z_ranges[i+1])
            
            if not np.any(mask):
                continue
                
            # Get pixels and depths in range
            pixels_in_range = img_channel[mask]
            depths_in_range = depths[mask]
            
            # Sort by intensity and select darkest pixels
            sorted_indices = np.argsort(pixels_in_range)
            n_select = min(int(np.ceil(fraction * len(pixels_in_range))), max_vals)
            
            for idx in sorted_indices[:n_select]:
                points.append((depths_in_range[idx], pixels_in_range[idx]))
        
        return np.array(points)
    
    @staticmethod
    def estimate_parameters(B_pts: np.ndarray, restarts: int = 10) -> Tuple[Callable, np.ndarray]:
        """
        Estimate backscatter model parameters using robust optimization.
        
        Args:
            B_pts: Array of (depth, intensity) pairs
            restarts: Number of random restarts for optimization
            
        Returns:
            Tuple of (backscatter function, parameters)
        """
        if len(B_pts) < 3:
            # Fallback to linear model if insufficient points
            return lambda d: 0.0, np.array([0.0, 0.0, 1.0])
        
        B_depths, B_vals = B_pts[:, 0], B_pts[:, 1]
        
        # Define bounds for parameters
        bounds_lower = [0, 0, 0.5]
        bounds_upper = [1, 5, 2]
        
        best_loss = np.inf
        best_params = None
        
        # Multiple random restarts for robustness
        for _ in range(restarts):
            try:
                # Random initialization
                p0 = bounds_lower + np.random.random(3) * (np.array(bounds_upper) - np.array(bounds_lower))
                
                # Curve fitting
                params, _ = curve_fit(
                    BackscatterModel.compute,
                    B_depths,
                    B_vals,
                    p0=p0,
                    bounds=(bounds_lower, bounds_upper),
                    maxfev=5000
                )
                
                # Compute loss
                loss = np.mean(np.abs(B_vals - BackscatterModel.compute(B_depths, *params)))
                
                if loss < best_loss:
                    best_loss = loss
                    best_params = params
                    
            except (RuntimeError, ValueError):
                continue
        
        if best_params is None:
            # Fallback to linear approximation
            slope, intercept = np.polyfit(B_depths, B_vals, 1)
            return lambda d: np.clip(slope * d + intercept, 0, 1), np.array([intercept, slope, 1.0])
        
        return lambda d: BackscatterModel.compute(d, *best_params), best_params


# ================================================================================
# Scattering Simulation
# ================================================================================

class ScatteringSimulator:
    """
    Monte Carlo simulation of light scattering in water using the 
    Henyey-Greenstein phase function.
    """
    
    @staticmethod
    def henyey_greenstein_phase_function(theta: np.ndarray, g: float) -> np.ndarray:
        """
        Compute the Henyey-Greenstein phase function.
        
        Args:
            theta: Scattering angle (radians)
            g: Asymmetry parameter (-1 to 1, typically 0.8 for water)
            
        Returns:
            Phase function values
        """
        return (1 - g**2) / (1 + g**2 - 2 * g * np.cos(theta))**1.5
    
    @staticmethod
    def monte_carlo_scattering(direct_signal: np.ndarray, depths: np.ndarray, 
                              scattering_coeff: float) -> np.ndarray:
        """
        Simulate forward scattering effects using Monte Carlo approach.
        
        Args:
            direct_signal: Direct transmission component
            depths: Depth map
            scattering_coeff: Scattering coefficient
            
        Returns:
            Scattered light estimation
        """
        # Ensure float type for numerical stability
        depths_float = depths.astype(np.float32)
        
        # Physical parameters
        g = 0.8  # Asymmetry factor for water
        absorption_coeff = 0.5
        energy_factor = 0.001
        
        # Adaptive kernel size based on scattering coefficient
        kernel_size = min(int(scattering_coeff * 5), 11)
        kernel_size = max(kernel_size, 3)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Create spatial kernel
        center = kernel_size // 2
        y, x = np.ogrid[-center:kernel_size-center, -center:kernel_size-center]
        r = np.sqrt(x**2 + y**2) + 1e-5
        
        # Compute scattering angles
        depth_mean = np.clip(np.mean(depths_float), 0.01, 10.0)
        theta = np.arctan(r / depth_mean)
        
        # Apply phase function
        phase_values = ScatteringSimulator.henyey_greenstein_phase_function(theta, g)
        kernel = phase_values / (np.sum(phase_values) + 1e-5)
        kernel *= energy_factor
        
        # Convolve with direct signal
        scattered_light = cv2.filter2D(direct_signal, -1, kernel)
        
        # Apply depth-dependent attenuation
        exponent = np.clip(-absorption_coeff * depths_float, -5, 0)
        depth_attenuation = np.exp(exponent)
        scattered_light *= depth_attenuation
        
        return scattered_light


# ================================================================================
# Illumination Estimation
# ================================================================================

class IlluminationEstimator:
    """
    Estimate the illumination map using neighborhood-based optimization
    with forward scattering compensation.
    """
    
    @staticmethod
    def estimate(img_channel: np.ndarray, backscatter: np.ndarray, 
                depths: np.ndarray, neighborhood_map: np.ndarray, 
                num_neighborhoods: int, scattering_coeff: float,
                p: float = 0.5, f: float = 2.0, 
                max_iters: int = 100, tol: float = 1e-5) -> np.ndarray:
        """
        Iteratively estimate illumination map.
        
        Args:
            img_channel: Single channel image
            backscatter: Backscatter component
            depths: Depth map
            neighborhood_map: Segmented depth neighborhoods
            num_neighborhoods: Number of neighborhoods
            scattering_coeff: Scattering coefficient
            p: Mixing parameter (0-1)
            f: Amplification factor
            max_iters: Maximum iterations
            tol: Convergence tolerance
            
        Returns:
            Estimated illumination map
        """
        # Initialize with direct transmission
        epsilon = 1e-6
        direct_signal = np.maximum(img_channel - backscatter, epsilon)
        
        # Prepare neighborhood information
        avg_illum = direct_signal.copy()
        avg_illum_neighbor = np.copy(avg_illum)
        
        # Precompute neighborhood locations and sizes
        locs_list = []
        sizes = []
        for label in range(1, num_neighborhoods + 1):
            locs = np.where(neighborhood_map == label)
            locs_list.append(locs)
            sizes.append(len(locs[0]))
        
        # Iterative refinement
        decay_factor = 0.95
        
        for iter_idx in range(max_iters):
            # Update neighborhood averages
            for i, (locs, size) in enumerate(zip(locs_list, sizes)):
                if size > 1:
                    neighborhood_sum = np.sum(avg_illum[locs])
                    avg_illum_neighbor[locs] = (neighborhood_sum - avg_illum[locs]) / (size - 1)
            
            # Incorporate forward scattering
            scattered = ScatteringSimulator.monte_carlo_scattering(
                direct_signal, depths, scattering_coeff
            )
            scattered_iter = scattered * (decay_factor ** iter_idx)
            
            # Update illumination estimate
            new_avg_illum = np.maximum(
                direct_signal * p + avg_illum_neighbor * (1 - p) + scattered_iter, 
                0
            )
            
            # Check convergence
            relative_change = np.max(np.abs(avg_illum - new_avg_illum) / (avg_illum + epsilon))
            if relative_change < tol:
                break
                
            avg_illum = new_avg_illum
        
        # Apply bilateral filtering for edge preservation
        return f * denoise_bilateral(avg_illum, sigma_color=0.1, sigma_spatial=15)


# ================================================================================
# Attenuation Estimation
# ================================================================================

class AttenuationEstimator:
    """
    Estimate wavelength-dependent attenuation coefficients using
    physical models and optimization.
    """
    
    # Water optical properties (absorption + scattering coefficients)
    WATER_PROPERTIES = {
        'I': {   # Clear ocean water
            'a': {450: 0.015, 550: 0.035, 650: 0.065},
            'b': {450: 0.032, 550: 0.020, 650: 0.010}
        },
        'II': {  # Coastal water
            'a': {450: 0.020, 550: 0.040, 650: 0.070},
            'b': {450: 0.050, 550: 0.040, 650: 0.030}
        },
        'III': { # Turbid harbor water
            'a': {450: 0.040, 550: 0.060, 650: 0.100},
            'b': {450: 0.080, 550: 0.070, 650: 0.050}
        }
    }
    
    @staticmethod
    def beta_model(depths: np.ndarray, a1: float, b1: float, 
                   a2: float, b2: float) -> np.ndarray:
        """
        Double-exponential model for depth-dependent attenuation.
        
        Args:
            depths: Depth values
            a1, b1, a2, b2: Model parameters
            
        Returns:
            Attenuation coefficients
        """
        return a1 * np.exp(-b1 * depths) + a2 * np.exp(-b2 * depths)
    
    @staticmethod
    def estimate_wideband_attenuation(depths: np.ndarray, illumination: np.ndarray,
                                     wavelength: int, water_type: str = 'II') -> np.ndarray:
        """
        Estimate attenuation coefficient map using optimization.
        
        Args:
            depths: Depth map
            illumination: Illumination map
            wavelength: Wavelength in nm
            water_type: Water type ('I', 'II', or 'III')
            
        Returns:
            Smoothed attenuation coefficient map
        """
        # Get theoretical coefficients
        water_props = AttenuationEstimator.WATER_PROPERTIES.get(water_type, 
                                                                AttenuationEstimator.WATER_PROPERTIES['II'])
        
        # Find closest wavelength
        available_wavelengths = list(water_props['a'].keys())
        closest_wavelength = min(available_wavelengths, key=lambda x: abs(x - wavelength))
        
        a_coeff = water_props['a'][closest_wavelength]
        b_coeff = water_props['b'][closest_wavelength]
        beta_theoretical = a_coeff + b_coeff
        
        # Prepare data
        depth_min = 0.1
        depths_safe = np.clip(depths, depth_min, None)
        illum_safe = np.clip(illumination, 0.01, 1.0)
        
        # Empirical estimation
        with np.errstate(divide='ignore', invalid='ignore'):
            beta_empirical = -np.log(illum_safe) / depths_safe
        beta_empirical = np.nan_to_num(beta_empirical, nan=beta_theoretical)
        beta_empirical = np.clip(beta_empirical, 0, 5.0)
        
        # Filter valid data points
        valid_mask = (illumination > 0.01) & (depths > depth_min)
        depths_valid = depths_safe[valid_mask]
        beta_valid = beta_empirical[valid_mask]
        
        if len(depths_valid) < 10:
            # Use theoretical model if insufficient data
            beta_final = beta_theoretical * np.ones_like(depths_safe)
        else:
            # Optimize double-exponential model
            def loss_func(params):
                beta_est = AttenuationEstimator.beta_model(depths_valid, *params)
                return np.mean((beta_valid - beta_est)**2)
            
            # Global optimization
            bounds = [(0, 10), (0, 1), (0, 10), (0, 1)]
            result = differential_evolution(loss_func, bounds, seed=42, maxiter=300)
            
            if result.success:
                beta_final = AttenuationEstimator.beta_model(depths_safe, *result.x)
            else:
                beta_final = beta_theoretical * np.ones_like(depths_safe)
        
        # Spatial smoothing with edge preservation
        beta_smoothed = cv2.medianBlur(beta_final.astype(np.float32), 5)
        beta_smoothed = cv2.bilateralFilter(beta_smoothed, 9, 75, 75)
        
        return beta_smoothed


# ================================================================================
# Image Enhancement
# ================================================================================

class UnderwaterImageEnhancer:
    """
    Post-processing enhancement for underwater images using
    adaptive white balance and contrast enhancement.
    """
    
    def __init__(self, clahe_clip_limit: float = 1.5,
                 clahe_grid_size: Tuple[int, int] = (8, 8),
                 gamma: float = 1.15):
        """
        Initialize the image enhancer.
        
        Args:
            clahe_clip_limit: CLAHE clip limit
            clahe_grid_size: CLAHE grid size
            gamma: Gamma correction value
        """
        self.clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=clahe_grid_size
        )
        self.gamma = gamma
    
    def apply_adaptive_white_balance(self, img: np.ndarray) -> np.ndarray:
        """
        Apply gray world white balance assumption.
        
        Args:
            img: Input image (RGB, 0-1 range)
            
        Returns:
            White balanced image
        """
        # Convert to LAB color space
        img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply gray world assumption
        l_mean = np.mean(l)
        a_mean = np.mean(a)
        b_mean = np.mean(b)
        
        # Adjust color channels
        a = a.astype(np.float32) - (a_mean - 128) * (l_mean / 128.0)
        b = b.astype(np.float32) - (b_mean - 128) * (l_mean / 128.0)
        
        # Clip and convert back
        a = np.clip(a, 0, 255).astype(np.uint8)
        b = np.clip(b, 0, 255).astype(np.uint8)
        
        balanced_lab = cv2.merge([l, a, b])
        balanced_rgb = cv2.cvtColor(balanced_lab, cv2.COLOR_LAB2RGB)
        
        return balanced_rgb.astype(np.float32) / 255.0
    
    def apply_clahe(self, img: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE to the luminance channel.
        
        Args:
            img: Input image (RGB, 0-1 range)
            
        Returns:
            Enhanced image
        """
        # Convert to LAB
        img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        l_clahe = self.clahe.apply(l)
        
        # Merge and convert back
        enhanced_lab = cv2.merge([l_clahe, a, b])
        enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        
        return enhanced_rgb.astype(np.float32) / 255.0
    
    def apply_gamma_correction(self, img: np.ndarray) -> np.ndarray:
        """
        Apply adaptive gamma correction.
        
        Args:
            img: Input image (RGB, 0-1 range)
            
        Returns:
            Gamma corrected image
        """
        # Apply gamma correction
        return np.power(np.clip(img, 0, 1), 1.0 / self.gamma)
    
    def enhance(self, img: np.ndarray) -> np.ndarray:
        """
        Apply full enhancement pipeline.
        
        Args:
            img: Input image (RGB, 0-1 range)
            
        Returns:
            Enhanced image
        """
        # Apply white balance
        balanced = self.apply_adaptive_white_balance(img)
        
        # Apply CLAHE
        clahe_result = self.apply_clahe(balanced)
        
        # Apply gamma correction
        gamma_result = self.apply_gamma_correction(balanced)
        
        # Blend results
        enhanced = 0.7 * clahe_result + 0.3 * gamma_result
        
        # Final contrast adjustment
        p2, p98 = np.percentile(enhanced, (2, 98))
        enhanced = exposure.rescale_intensity(enhanced, in_range=(p2, p98))
        
        return np.clip(enhanced, 0, 1)


# ================================================================================
# Neighborhood Map Construction
# ================================================================================

class NeighborhoodMapper:
    """
    Construct and refine depth-based neighborhood maps for spatial processing.
    """
    
    @staticmethod
    def construct_neighborhood_map(depths: np.ndarray, epsilon: float = 0.05) -> Tuple[np.ndarray, int]:
        """
        Segment depth map into neighborhoods using region growing.
        
        Args:
            depths: Depth map
            epsilon: Relative depth tolerance
            
        Returns:
            Tuple of (neighborhood map, number of neighborhoods)
        """
        eps = (np.max(depths) - np.min(depths)) * epsilon
        nmap = np.zeros_like(depths, dtype=np.int32)
        n_neighborhoods = 0
        
        # Region growing
        while np.any(nmap == 0):
            # Find unassigned pixel
            locs_y, locs_x = np.where(nmap == 0)
            start_idx = np.random.randint(0, len(locs_x))
            start_y, start_x = locs_y[start_idx], locs_x[start_idx]
            
            n_neighborhoods += 1
            
            # BFS for region growing
            queue = collections.deque([(start_y, start_x)])
            start_depth = depths[start_y, start_x]
            
            while queue:
                y, x = queue.popleft()
                
                if nmap[y, x] != 0:
                    continue
                
                if np.abs(depths[y, x] - start_depth) <= eps:
                    nmap[y, x] = n_neighborhoods
                    
                    # Add neighbors
                    for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < depths.shape[0] and 0 <= nx < depths.shape[1]:
                            if nmap[ny, nx] == 0:
                                queue.append((ny, nx))
        
        return nmap, n_neighborhoods
    
    @staticmethod
    def refine_neighborhood_map(nmap: np.ndarray, min_size: int = 50, 
                               radius: int = 3) -> Tuple[np.ndarray, int]:
        """
        Refine neighborhood map by merging small regions.
        
        Args:
            nmap: Initial neighborhood map
            min_size: Minimum region size
            radius: Morphological closing radius
            
        Returns:
            Tuple of (refined map, number of neighborhoods)
        """
        # Count region sizes
        unique_labels, counts = np.unique(nmap[nmap > 0], return_counts=True)
        
        # Relabel large regions
        refined_nmap = np.zeros_like(nmap)
        new_label = 1
        
        for label, size in zip(unique_labels, counts):
            if size >= min_size:
                refined_nmap[nmap == label] = new_label
                new_label += 1
        
        # Merge small regions into nearest neighbors
        for label, size in zip(unique_labels, counts):
            if size < min_size and label > 0:
                mask = (nmap == label)
                # Find nearest labeled pixel
                for y, x in zip(*np.where(mask)):
                    refined_nmap[y, x] = NeighborhoodMapper._find_nearest_label(
                        refined_nmap, y, x
                    )
        
        # Morphological closing
        refined_nmap = closing(refined_nmap, square(radius))
        
        return refined_nmap, new_label - 1
    
    @staticmethod
    def _find_nearest_label(nmap: np.ndarray, y: int, x: int) -> int:
        """Find nearest non-zero label using BFS."""
        visited = set()
        queue = collections.deque([(y, x)])
        
        while queue:
            cy, cx = queue.popleft()
            
            if (cy, cx) in visited:
                continue
            visited.add((cy, cx))
            
            if 0 <= cy < nmap.shape[0] and 0 <= cx < nmap.shape[1]:
                if nmap[cy, cx] > 0:
                    return nmap[cy, cx]
                
                for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    queue.append((cy + dy, cx + dx))
        
        return 0


# ================================================================================
# Main Pipeline
# ================================================================================

class UnderwaterImageRestoration:
    """
    Main pipeline for underwater image restoration using the improved SeaThru algorithm.
    """
    
    def __init__(self, water_type: str = 'II'):
        """
        Initialize the restoration pipeline.
        
        Args:
            water_type: Type of water ('I', 'II', or 'III')
        """
        self.water_type = water_type
        self.enhancer = UnderwaterImageEnhancer()
        
    def preprocess_depth_map(self, depths: np.ndarray) -> np.ndarray:
        """
        Smooth and preprocess depth map.
        
        Args:
            depths: Raw depth map
            
        Returns:
            Processed depth map
        """
        # Edge-preserving smoothing
        return cv2.edgePreservingFilter(
            depths.astype(np.float32), 
            flags=1, 
            sigma_s=5, 
            sigma_r=0.1
        )
    
    def recover_image(self, img: np.ndarray, depths: np.ndarray, 
                     backscatter: np.ndarray, beta: np.ndarray) -> np.ndarray:
        """
        Recover the scene radiance from the underwater image.
        
        Args:
            img: Input underwater image
            depths: Depth map
            backscatter: Estimated backscatter
            beta: Attenuation coefficients
            
        Returns:
            Recovered image
        """
        # Compute transmission
        max_exponent = 5
        exponent = np.clip(beta * np.expand_dims(depths, axis=2), 0, max_exponent)
        transmission = np.exp(-exponent)
        
        # Remove backscatter and correct attenuation
        epsilon = 1e-6
        direct_signal = (img - backscatter) / (transmission + epsilon)
        
        # Normalize and enhance
        direct_signal = np.clip(direct_signal, 0, None)
        
        # Scale to 0-1 range
        for c in range(3):
            channel = direct_signal[:, :, c]
            p1, p99 = np.percentile(channel, (1, 99))
            if p99 > p1:
                direct_signal[:, :, c] = (channel - p1) / (p99 - p1)
        
        direct_signal = np.clip(direct_signal, 0, 1)
        
        # Apply enhancement
        return self.enhancer.enhance(direct_signal)
    
    def estimate_scattering_coefficient(self, img_channel: np.ndarray, 
                                      depths: np.ndarray, wavelength: int) -> float:
        """
        Estimate scattering coefficient for a given channel.
        
        Args:
            img_channel: Single channel image
            depths: Depth map
            wavelength: Wavelength in nm
            
        Returns:
            Estimated scattering coefficient
        """
        # Get water properties
        water_props = AttenuationEstimator.WATER_PROPERTIES.get(
            self.water_type, 
            AttenuationEstimator.WATER_PROPERTIES['II']
        )
        
        # Find closest wavelength
        available_wavelengths = list(water_props['b'].keys())
        closest_wavelength = min(available_wavelengths, key=lambda x: abs(x - wavelength))
        
        # Return theoretical scattering coefficient
        return water_props['b'][closest_wavelength]
    
    def process(self, img: np.ndarray, depths: np.ndarray, 
                p: float = 0.5, f: float = 2.0) -> np.ndarray:
        """
        Process underwater image using the full restoration pipeline.
        
        Args:
            img: Input underwater image (RGB, 0-1 range)
            depths: Depth map
            p: Illumination mixing parameter
            f: Illumination amplification factor
            
        Returns:
            Restored image
        """
        # Wavelength mapping
        wavelengths = {'R': 650, 'G': 550, 'B': 450}
        
        # Preprocess depth map
        print("Preprocessing depth map...", flush=True)
        depths = self.preprocess_depth_map(depths)
        
        # Estimate backscatter for each channel
        print("Estimating backscatter...", flush=True)
        backscatter_funcs = []
        backscatter_params = []
        
        for c, (channel_name, wavelength) in enumerate(wavelengths.items()):
            # Find backscatter points
            B_pts = BackscatterModel.find_estimation_points(img[:, :, c], depths)
            
            # Estimate parameters
            B_func, B_params = BackscatterModel.estimate_parameters(B_pts)
            backscatter_funcs.append(B_func)
            backscatter_params.append(B_params)
            
            print(f"  {channel_name} channel: B_inf={B_params[0]:.3f}, "
                  f"beta_B={B_params[1]:.3f}, gamma={B_params[2]:.3f}")
        
        # Construct neighborhood map
        print("Constructing neighborhood map...", flush=True)
        nmap, n_neighborhoods = NeighborhoodMapper.construct_neighborhood_map(depths, epsilon=0.15)
        nmap, n_neighborhoods = NeighborhoodMapper.refine_neighborhood_map(nmap, min_size=50)
        print(f"  Found {n_neighborhoods} neighborhoods")
        
        # Estimate illumination for each channel
        print("Estimating illumination...", flush=True)
        illumination = np.zeros_like(img)
        
        for c, (channel_name, wavelength) in enumerate(wavelengths.items()):
            # Get scattering coefficient
            scatter_coeff = self.estimate_scattering_coefficient(img[:, :, c], depths, wavelength)
            
            # Compute backscatter
            B_c = backscatter_funcs[c](depths)
            
            # Estimate illumination
            illumination[:, :, c] = IlluminationEstimator.estimate(
                img[:, :, c], B_c, depths, nmap, n_neighborhoods,
                scatter_coeff, p=p, f=f
            )
        
        # Estimate attenuation coefficients
        print("Estimating attenuation coefficients...", flush=True)
        beta = np.zeros((depths.shape[0], depths.shape[1], 3))
        
        for c, (channel_name, wavelength) in enumerate(wavelengths.items()):
            beta[:, :, c] = AttenuationEstimator.estimate_wideband_attenuation(
                depths, illumination[:, :, c], wavelength, self.water_type
            )
        
        # Compute final backscatter
        backscatter = np.zeros_like(img)
        for c in range(3):
            backscatter[:, :, c] = backscatter_funcs[c](depths)
        
        # Recover image
        print("Recovering image...", flush=True)
        recovered = self.recover_image(img, depths, backscatter, beta)
        
        return recovered


# ================================================================================
# Utility Functions
# ================================================================================

def load_image_and_depth_map(img_path: str, depth_path: str, 
                           max_size: int = 1024) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and preprocess image and depth map.
    
    Args:
        img_path: Path to image file
        depth_path: Path to depth map file
        max_size: Maximum image dimension
        
    Returns:
        Tuple of (image, depth_map) both as numpy arrays
    """
    # Load image
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not load image from {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Load depth map
    depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
    if depth is None:
        raise ValueError(f"Could not load depth map from {depth_path}")
    
    # Resize if necessary
    h, w = img.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        depth = cv2.resize(depth, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Normalize
    img = img.astype(np.float32) / 255.0
    depth = depth.astype(np.float32) / 255.0
    
    return img, depth


def scale(img):
    """
    Scale image to 0-1 range.
    
    This function is provided for compatibility with the original codebase.
    
    Args:
        img: Input image as numpy array
        
    Returns:
        Scaled image in 0-1 range
    """
    img_float = img.astype(np.float32)
    min_val = np.min(img_float)
    max_val = np.max(img_float)
    
    if max_val - min_val < 1e-6:
        # If image is uniform, return middle gray
        return np.full_like(img_float, 0.5)
    
    return (img_float - min_val) / (max_val - min_val)


def getUCIQE(img):
    """
    Calculate UCIQE (Underwater Color Image Quality Evaluation) metric.
    
    This is a placeholder function. In a full implementation, this would
    calculate the UCIQE metric as described in:
    Yang, M., & Sowmya, A. (2015). An underwater color image quality 
    evaluation metric. IEEE Transactions on Image Processing.
    
    Args:
        img: Input image (0-1 range or 0-255 range)
        
    Returns:
        UCIQE score (higher is better)
    """
    # Ensure image is in 0-255 range
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    
    # Placeholder implementation
    # In practice, this would calculate:
    # - Chroma variance (σc)
    # - Luminance contrast (conl)
    # - Average saturation (μs)
    # UCIQE = c1 * σc + c2 * conl + c3 * μs
    
    return 0.5  # Placeholder value


# ================================================================================
# Legacy Interface for Backward Compatibility
# ================================================================================

def run_pipeline(img: np.ndarray, depths: np.ndarray, args) -> np.ndarray:
    """
    Legacy interface function for backward compatibility with existing code.
    
    This function provides the same interface as the original improved_seathru.py
    to ensure seamless integration with existing pipelines.
    
    Args:
        img: Input underwater image (RGB, 0-1 range, float32/float64)
        depths: Depth map (0-1 range, float32/float64)
        args: Namespace object with attributes:
            - p: Illumination estimation mixing parameter (default: 0.5)
            - f: Illumination amplification factor (default: 2.0)
            - water_type: Water type ('I', 'II', or 'III') (optional, default: 'II')
            - min_depth: Minimum depth for backscatter estimation (optional)
            - max_depth: Maximum depth for backscatter estimation (optional)
            
    Returns:
        Enhanced image as numpy array (RGB, 0-1 range)
    """
    # Input validation and normalization
    if img is None or depths is None:
        raise ValueError("Input image and depth map cannot be None")
    
    # Ensure correct data types
    img = img.astype(np.float32) if img.dtype != np.float32 else img
    depths = depths.astype(np.float32) if depths.dtype != np.float32 else depths
    
    # Ensure inputs are in 0-1 range
    if img.max() > 1.0:
        print("Warning: Input image appears to be in 0-255 range, normalizing to 0-1")
        img = img / 255.0
    
    if depths.max() > 1.0:
        print("Warning: Depth map appears to be in 0-255 range, normalizing to 0-1")
        depths = depths / 255.0
    
    # Validate image dimensions
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Input image must be RGB with shape (H, W, 3), got {img.shape}")
    
    if depths.ndim != 2:
        raise ValueError(f"Depth map must be 2D with shape (H, W), got {depths.shape}")
    
    if img.shape[:2] != depths.shape:
        raise ValueError(f"Image shape {img.shape[:2]} doesn't match depth shape {depths.shape}")
    
    # Extract parameters from args
    p = getattr(args, 'p', 0.5)
    f = getattr(args, 'f', 2.0)
    water_type = getattr(args, 'water_type', 'II')
    
    # Validate parameters
    p = np.clip(p, 0.0, 1.0)
    f = max(0.1, f)  # Ensure f is positive
    
    if water_type not in ['I', 'II', 'III']:
        print(f"Warning: Unknown water type '{water_type}', using 'II' (coastal)")
        water_type = 'II'
    
    try:
        # Create restoration pipeline
        restorer = UnderwaterImageRestoration(water_type=water_type)
        
        # Process image
        recovered = restorer.process(img, depths, p=p, f=f)
        
        # Ensure output is in correct format
        recovered = np.clip(recovered, 0, 1).astype(np.float32)
        
        return recovered
        
    except Exception as e:
        print(f"Error in underwater image restoration: {str(e)}")
        print("Returning original image")
        return img


# ================================================================================
# Main Entry Point
# ================================================================================

def main():
    """
    Main function for command-line usage.
    """
    import argparse
    import matplotlib.pyplot as plt
    
    parser = argparse.ArgumentParser(
        description='Underwater Image Enhancement using Improved SeaThru Algorithm'
    )
    parser.add_argument('--image', type=str, required=True, 
                       help='Path to input underwater image')
    parser.add_argument('--depth', type=str, required=True, 
                       help='Path to depth map')
    parser.add_argument('--output', type=str, required=True, 
                       help='Path to output enhanced image')
    parser.add_argument('--water-type', type=str, default='II', 
                       choices=['I', 'II', 'III'],
                       help='Water type: I (clear), II (coastal), III (turbid)')
    parser.add_argument('--p', type=float, default=0.5, 
                       help='Illumination estimation mixing parameter')
    parser.add_argument('--f', type=float, default=2.0, 
                       help='Illumination amplification factor')
    
    args = parser.parse_args()
    
    # Load data
    print("Loading image and depth map...")
    img, depths = load_image_and_depth_map(args.image, args.depth)
    
    # Method 1: Direct class usage (recommended for new code)
    restorer = UnderwaterImageRestoration(water_type=args.water_type)
    recovered = restorer.process(img, depths, p=args.p, f=args.f)
    
    # Method 2: Legacy interface (for backward compatibility)
    # recovered = run_pipeline(img, depths, args)
    
    # Save result
    plt.imsave(args.output, recovered)
    print(f"Enhanced image saved to {args.output}")
    
    # Display comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    axes[1].imshow(depths, cmap='gray')
    axes[1].set_title('Depth Map')
    axes[1].axis('off')
    
    axes[2].imshow(recovered)
    axes[2].set_title('Enhanced')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()