# -*- coding: utf-8 -*-
"""
Adapted from @author: Ben A Duffy - University of Southern California
"""

from typing import Sequence

import numpy as np
import math

try:
    import finufftpy

    finufft = True
except ImportError:
    finufft = False


def get_center_rect(im_shape, pct=0.07, dim=0):
    """get center 'rectangle' covering pct percent of a 2d or 3d array"""
    mask = np.zeros(im_shape)
    half_pct = (pct / 2)
    center = [int(x / 2) for x in im_shape]
    mask = np.swapaxes(mask, 0, dim)
    mask[:, center[1] - math.ceil(im_shape[1] * half_pct):math.ceil(center[1] + im_shape[1] * half_pct)] = 1
    mask = np.swapaxes(mask, 0, dim)
    return mask


def get_center_cross(im_shape, pct=0.07):
    """get center 'cross' covering pct percent of a 2d or 3d array"""
    mask = np.zeros(im_shape)
    half_pct = (pct / 2)
    center = [int(x / 2) for x in im_shape]

    if len(im_shape) == 3:
        mask[center[0] - math.ceil(im_shape[0] * half_pct):math.ceil(center[0] + im_shape[0] * half_pct), :, :] = 1
        mask[:, center[1] - math.ceil(im_shape[1] * half_pct):math.ceil(center[1] + im_shape[1] * half_pct), :] = 1
        mask[:, :, center[2] - math.ceil(im_shape[2] * half_pct):math.ceil(center[2] + im_shape[2] * half_pct)] = 1

    elif len(im_shape) == 2:
        mask[:, center[1] - math.ceil(im_shape[1] * half_pct):math.ceil(center[1] + im_shape[1] * half_pct)] = 1
        mask[center[0] - math.ceil(im_shape[0] * half_pct):math.ceil(center[0] + im_shape[0] * half_pct), :] = 1

    return mask


def segment_array_by_locs(shape: Sequence[int], locations: Sequence[int]):
    """
    generate array segmented by locations
    :param shape: shape of array
    :param locations: locations
    :return: segmentation mask
    """
    mask_out = np.zeros(np.prod(shape), dtype=int)
    for i in range(len(locations) - 1):
        l = [locations[i],
             locations[i + 1]]
        mask_out[l[0]:l[1]] = i + 1
    return mask_out.reshape(shape)


def assign_segments_to_random_indices(shape: Sequence[int], seg_lengths: Sequence[int]) -> np.ndarray:
    """assign segments to random locations and return segmentation mask"""

    seg_mask = np.zeros(shape, dtype='int')
    random_indices = sorted(np.random.choice(shape, replace=False, size=sum(seg_lengths)))

    seg_new_indices = (np.cumsum(seg_lengths)).tolist()

    seg_new_indices = [0] + seg_new_indices

    for i in range(len(seg_new_indices) - 1):
        seg_mask[random_indices[seg_new_indices[i]:seg_new_indices[i + 1]]] = i + 1
    return seg_mask


def assign_segments_to_random_blocks(shape: Sequence[int], seg_lengths: Sequence[int]) -> np.ndarray:
    """
    generate randomly segmented array based on seg_lengths
    :param shape - image shape
    :param seg_lengths -
    :return: segmentation label image
    """
    seg_mask = np.zeros(shape, dtype='int')
    seg_lengths_sorted = sorted(seg_lengths, reverse=True)
    for i, seg_len in enumerate(seg_lengths_sorted):
        loc = np.random.randint(0, seg_mask.size)
        while (sum(seg_mask[loc:loc + seg_len]) != 0) or (loc + seg_len > seg_mask.size):
            loc = np.random.randint(0, seg_mask.size)
        seg_mask[loc:loc + seg_len] = i + 1
    return seg_mask


def create_rand_partition(im_length: int, n_seg: int):
    """
    :param im_length: length of 1D array to partition
    :param n_seg: num segs to partition into
    :return: partition locations (list of indices)
    """
    rand_segment_locs = sorted(np.random.randint(im_length, size=n_seg + 1).astype(list))
    rand_segment_locs[0] = 0
    rand_segment_locs[-1] = None
    return rand_segment_locs


def create_rotation_matrix_3d(angles: Sequence[float]) -> np.ndarray:
    """
    given a list of 3 angles, create a 3x3 rotation matrix that describes rotation about the origin
    :param angles: rotation angles in 3 dimensions
    :return (np.array) : rotation matrix 3x3
    """

    mat1 = np.array([[1., 0., 0.],
                     [0., math.cos(angles[0]), math.sin(angles[0])],
                     [0., -math.sin(angles[0]), math.cos(angles[0])]],
                    dtype='float')

    mat2 = np.array([[math.cos(angles[1]), 0., -math.sin(angles[1])],
                     [0., 1., 0.],
                     [math.sin(angles[1]), 0., math.cos(angles[1])]],
                    dtype='float')

    mat3 = np.array([[math.cos(angles[2]), math.sin(angles[2]), 0.],
                     [-math.sin(angles[2]), math.cos(angles[2]), 0.],
                     [0., 0., 1.]],
                    dtype='float')

    mat = (mat1 @ mat2) @ mat3
    return mat


def _translate_freq_domain(freq_domain: np.ndarray, translations: np.ndarray):
    """
    image domain translation by adding phase shifts in frequency domain
    :param freq_domain - frequency domain data 3d
    :param translations - np.array of shape (3, self.num_phase_encoding_steps)
    :return frequency domain array with phase shifts added according to translations:
    """

    lin_spaces = [np.linspace(-0.5, 0.5, x) for x in freq_domain.shape]
    meshgrids = np.meshgrid(*lin_spaces, indexing='ij')
    grid_coords = np.array([mg.flatten() for mg in meshgrids])

    phase_shift = np.multiply(grid_coords, translations).sum(axis=0)  # phase shift is added
    exp_phase_shift = np.exp(-2j * math.pi * phase_shift)
    freq_domain_translated = np.multiply(exp_phase_shift, freq_domain.flatten(order='C')).reshape(freq_domain.shape)

    return freq_domain_translated


def _fft_im(image: np.ndarray) -> np.ndarray:
    output = (np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(image)))).astype(np.complex128)
    return output


def _ifft_im(freq_domain: np.ndarray) -> np.ndarray:
    output = np.fft.ifftshift(np.fft.ifftn(freq_domain))
    return output


class MotionSimLayer:
    """given a real valued 3D MRI image, simulate random translations and rotations

        :param std_rotation_angle (float) : std of rotations
        :param std_translation (float) : std of translations
        :param corrupt_pct_range (list of ints): range of percents
        :param freq_encoding_dim (list of ints): randomly choose freq encoding dim
        :param preserve_center_pct (float): percentage of k-space center to preserve
        :param name (str) : name of layer
        :param apply_mask (bool): apply mask to output or not
        :param nufft (bool): whether to use nufft for introducing rotations
        :param num_pieces (int): number of pieces for piecewise constant simulation
        :param corruption_scheme 'piecewise_transient', 'piecewise_constant', 'gaussian'

       raises ImportError if nufft is true but finufft cannot be imported

    """
    
    def __init__(self, std_rotation_angle=0, std_translation=10, corrupt_pct_range=(15, 20),
                 freq_encoding_dim=(0, 1, 2), preserve_center_pct=0.07,
                 apply_mask=True, nufft=False, corruption_scheme='piecewise_transient',
                 n_seg=8, fixed_n_seg=False):

        self.trajectory = None
        self.preserve_center_frequency_pct = preserve_center_pct
        self.freq_encoding_choice = freq_encoding_dim
        self.frequency_encoding_dim = np.random.choice(self.freq_encoding_choice)
        self.std_rotation_angle, self.std_translation = std_rotation_angle, std_translation
        self.corrupt_pct_range = corrupt_pct_range
        self.apply_mask = apply_mask
        self.corruption_scheme = corruption_scheme
        self.n_seg = n_seg
        self.fixed_n_seg = fixed_n_seg
        self.nufft = nufft
        self.transformation_params = dict()

        if (not finufft) and nufft:
            raise ImportError('finufftpy cannot be imported')
        
    def _calc_dimensions(self, im_shape):
        """
        calculate dimensions based on im_shape
        :param im_shape (list/tuple) : image shape

        - sets self.phase_encoding_dims, self.phase_encoding_shape, self.num_phase_encoding_steps, self.frequency_encoding_dim

        """
        pe_dims = [0, 1, 2]
        pe_dims.pop(self.frequency_encoding_dim)
        self.phase_encoding_dims = pe_dims
        im_shape = list(im_shape)
        self.im_shape = im_shape.copy()
        im_shape.pop(self.frequency_encoding_dim)
        self.phase_encoding_shape = im_shape
        self.num_phase_encoding_steps = self.phase_encoding_shape[0] * self.phase_encoding_shape[1]
        self.frequency_encoding_dim = len(
            self.im_shape) - 1 if self.frequency_encoding_dim == -1 else self.frequency_encoding_dim

    def _simulate_random_trajectory(self):
        """
        corruption_scheme is either {'piecewise_transient','piecewise_constant','gaussian'}
        simulates transient blocked random trajectory using a random number of lines generated from corrupt_pct_range
        returns translations and rotations
        """

        pct_corrupt = np.random.uniform(*[x / 100 for x in self.corrupt_pct_range])

        corrupt_matrix_shape = [int(x * math.sqrt(pct_corrupt)) for x in self.phase_encoding_shape]
        # TODO keep this as a vector

        if np.prod(corrupt_matrix_shape) == 0:
            corrupt_matrix_shape = [1, 1]

        if self.corruption_scheme in {'gaussian'}:
            n_seg = np.prod(corrupt_matrix_shape)

        else:
            if self.fixed_n_seg:
                n_seg = self.n_seg
            else:
                n_seg = np.random.choice(np.arange(1, self.n_seg))

        # segment a smaller vector occupying pct_corrupt percent of the space
        if self.corruption_scheme in {'piecewise_transient', 'piecewise_constant'}:
            seg_locs = create_rand_partition(np.prod(corrupt_matrix_shape), n_seg=n_seg)
        else:
            seg_locs = list(range(n_seg))

        rand_segmentation = segment_array_by_locs(
            shape=np.prod(corrupt_matrix_shape), locations=seg_locs)

        seg_lengths = [(rand_segmentation == seg_num).sum() for seg_num in np.unique(rand_segmentation)]

        # assign segments to a vector with same number of elements as pe-steps
        if self.corruption_scheme in {'piecewise_transient', 'gaussian'}:
            seg_vector = assign_segments_to_random_indices(np.prod(self.phase_encoding_shape), seg_lengths)
        else:
            seg_vector = assign_segments_to_random_blocks(np.prod(self.phase_encoding_shape), seg_lengths)

        # reshape to phase encoding shape with a random order
        reshape_order = np.random.choice(['F', 'C'])
        seg_array = seg_vector.reshape(self.phase_encoding_shape, order=reshape_order)
        self.order = reshape_order

        # mask center k-space
        if reshape_order == 'C':
            mask_not_including_center = get_center_rect(self.phase_encoding_shape, dim=1) == 0
        else:
            mask_not_including_center = get_center_rect(self.phase_encoding_shape, dim=0) == 0
            
        seg_array = seg_array * mask_not_including_center

        # generate random translations and rotations
        rand_translations = np.random.normal(scale=self.std_translation, size=(n_seg + 1, 3))
        rand_rotations = np.random.normal(scale=self.std_rotation_angle, size=(n_seg + 1, 3))

        # if segment==0, then no motion
        rand_translations[0, :] = 0
        rand_rotations[0, :] = 0

        # lookup values for each segment
        translations_pe = [rand_translations[:, i][seg_array] for i in range(3)]
        rotations_pe = [rand_rotations[:, i][seg_array] for i in range(3)]

        # reshape and convert to radians
        translations = np.array(
            [np.broadcast_to(np.expand_dims(x, self.frequency_encoding_dim), self.im_shape) for x in translations_pe])
        rotations = np.array(
            [np.broadcast_to(np.expand_dims(x, self.frequency_encoding_dim), self.im_shape) for x in rotations_pe])

        rotations = rotations * (math.pi / 180.)  # convert to radians

        translations = translations.reshape(3, -1)
        rotations = rotations.reshape(3, -1).reshape(3, -1)

        return translations, rotations

    def _rotate_coordinates(self):
        """
        :return: grid_coordinates after applying self.rotations
        """
        center = [math.ceil((x - 1) / 2) for x in self.im_shape]

        [i1, i2, i3] = np.meshgrid(np.arange(self.im_shape[0]) - center[0],
                                   np.arange(self.im_shape[1]) - center[1],
                                   np.arange(self.im_shape[2]) - center[2], indexing='ij')

        grid_coordinates = np.array([i1.T.flatten(), i2.T.flatten(), i3.T.flatten()])

        rotations = self.rotations.reshape([3] + self.im_shape)
        ix = (len(self.im_shape) + 1) * [slice(None)]
        ix[self.frequency_encoding_dim + 1] = 0  # dont need to rotate along freq encoding

        rotations = rotations[ix].reshape(3, -1)
        rotation_matrices = np.apply_along_axis(create_rotation_matrix_3d, axis=0, arr=rotations).transpose([-1, 0, 1])
        rotation_matrices = rotation_matrices.reshape(self.phase_encoding_shape + [3, 3])
        rotation_matrices = np.expand_dims(rotation_matrices, self.frequency_encoding_dim)

        rotation_matrices = np.tile(rotation_matrices,
                                    reps=([self.im_shape[
                                               self.frequency_encoding_dim] if i == self.frequency_encoding_dim else 1
                                           for i in range(5)]))  # tile in freq encoding dimension

        rotation_matrices = rotation_matrices.reshape([-1, 3, 3])

        # tile grid coordinates
        grid_coordinates_tiled = np.tile(grid_coordinates, [3, 1])
        grid_coordinates_tiled = grid_coordinates_tiled.reshape([3, -1], order='F').T
        rotation_matrices = rotation_matrices.reshape([-1, 3])

        new_grid_coords = (rotation_matrices * grid_coordinates_tiled).sum(axis=1)

        # reshape new grid coords back to 3 x nvoxels
        new_grid_coords = new_grid_coords.reshape([3, -1], order='F')

        # scale data between -pi and pi
        max_vals = [abs(x) for x in grid_coordinates[:, 0]]
        new_grid_coordinates_scaled = [(new_grid_coords[i, :] / max_vals[i]) * math.pi for i in
                                       range(new_grid_coords.shape[0])]
        new_grid_coordinates_scaled = [np.asfortranarray(i) for i in new_grid_coordinates_scaled]

        return new_grid_coordinates_scaled, [grid_coordinates, new_grid_coords]

    def _nufft(self, freq_domain_data, iflag=1, eps=1E-7):
        """
        rotate coordinates and perform nufft
        :param freq_domain_data:
        :param iflag/eps: see finufftpy doc
        :param eps: precision of nufft
        :return: nufft of freq_domain_data after applying self.rotations
        """

        if not finufft:
            raise ImportError('finufftpy not available')

        new_grid_coords = self._rotate_coordinates()[0]

        # initialize array for nufft output
        f = np.zeros([len(new_grid_coords[0])], dtype=np.complex128, order='F')

        freq_domain_data_flat = np.asfortranarray(freq_domain_data.flatten(order='F'))

        finufftpy.nufft3d1(new_grid_coords[0], new_grid_coords[1], new_grid_coords[2], freq_domain_data_flat,
                           iflag, eps, self.im_shape[0], self.im_shape[1],
                           self.im_shape[2], f, debug=0, spread_debug=0, spread_sort=2, fftw=0, modeord=0,
                           chkbnds=0, upsampfac=1.25)  # upsampling at 1.25 saves time at low precisions

        im_out = f.reshape(self.im_shape, order='F')

        return im_out

    def layer_op(self, input_image, translations_rotations=None):

        self.frequency_encoding_dim = np.random.choice(self.freq_encoding_choice)

        original_image = input_image
        self._calc_dimensions(original_image.shape)

        if translations_rotations is None:
            translations, rotations = self._simulate_random_trajectory()
        else:
            translations, rotations = translations_rotations

        self.transformation_params['translations'] = translations
        self.transformation_params['rotations'] = rotations

        # fft
        im_freq_domain = _fft_im(original_image)
        translated_im_freq_domain = _translate_freq_domain(freq_domain=im_freq_domain, translations=translations)

        # iNufft for rotations
        if self.nufft:
            corrupted_im = self._nufft(translated_im_freq_domain)
            corrupted_im = corrupted_im / corrupted_im.size  # normalize

        else:
            corrupted_im = _ifft_im(translated_im_freq_domain)

        # magnitude
        corrupted_im = abs(corrupted_im)

        return corrupted_im

    def __call__(self, *args, **kwargs):
        return self.layer_op(*args, **kwargs)


"""PERFORM MOTION SIMULATION"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images

def apply_motion_simulation(images, ms_layer):
    simulated_images = []
    for img in images:
        output = ms_layer.layer_op(img)
        simulated_images.append(output)
    return simulated_images

def save_simulated_images(simulated_images, original_folder, output_folder):
    for i, sim_img in enumerate(simulated_images):
        filename = os.listdir(original_folder)[i]
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, sim_img)

def create_output_folder(output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

def plot_im(im, title=None, *args, **kwargs):
    plt.imshow(im, *args, **kwargs)
    plt.axis('off')
    if title:
        plt.title(title)

base_directory = 'path/to/base_directory'
subdirectory = 'path/to/subdirectory'

full_path = os.path.join(base_directory, subdirectory)

if __name__ == '__main__':  
    input_folders = [
        os.path.join(full_path, 'test', 'T1_gt'),
        os.path.join(full_path, 'test', 'T1CE_gt'),
        os.path.join(full_path, 'test', 'T2_gt'),
        os.path.join(full_path, 'test', 'FL_gt'),
        os.path.join(full_path, 'train', 'T1_gt'),
        os.path.join(full_path, 'train', 'T1CE_gt'),
        os.path.join(full_path, 'train', 'T2_gt'),
        os.path.join(full_path, 'train', 'FL_gt'),
        os.path.join(full_path, 'valid', 'T1_gt'),
        os.path.join(full_path, 'valid', 'T1CE_gt'),
        os.path.join(full_path, 'valid', 'T2_gt'),
        os.path.join(full_path, 'valid', 'FL_gt')
    ]

    output_folders = [
        os.path.join(full_path, 'test', 'T1_motion'),
        os.path.join(full_path, 'test', 'T1CE_motion'),
        os.path.join(full_path, 'test', 'T2_motion'),
        os.path.join(full_path, 'test', 'FL_motion'),
        os.path.join(full_path, 'train', 'T1_motion'),
        os.path.join(full_path, 'train', 'T1CE_motion'),
        os.path.join(full_path, 'train', 'T2_motion'),
        os.path.join(full_path, 'train', 'FL_motion'),
        os.path.join(full_path, 'valid', 'T1_motion'),
        os.path.join(full_path, 'valid', 'T1CE_motion'),
        os.path.join(full_path, 'valid', 'T2_motion'),
        os.path.join(full_path, 'valid', 'FL_motion')
    ]

    for input_folder, output_folder in zip(input_folders, output_folders):
        create_output_folder(output_folder)

        images = load_images_from_folder(input_folder)
 
        ms_layer = MotionSimLayer(corrupt_pct_range=[25, 30],
                                             corruption_scheme='piecewise_constant',
                                             n_seg=8)

        simulated_images = apply_motion_simulation(images, ms_layer)

        save_simulated_images(simulated_images, input_folder, output_folder)
        print('Simulation completed.')
