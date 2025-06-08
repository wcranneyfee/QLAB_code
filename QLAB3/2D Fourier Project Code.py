import numpy as np
from scipy.fft import fft2, fftshift, fftn
import matplotlib.pyplot as plt
from clathrate import Clathrate

def FFT_2D(arr, fig_num):

    fft_arr = fftshift(fft2(arr))

    plt.figure(fig_num)
    plt.imshow(arr)
    plt.xlabel('x')
    plt.ylabel('y')

    plt.figure(fig_num + 1)
    mags = np.abs(fft_arr)
    plt.imshow(mags)
    plt.xlabel('kx')
    plt.ylabel('ky')


def FFT_3D(arr, fig_num, min_intensity, center_slice=True):

    fig = plt.figure(fig_num)
    ax = fig.add_subplot(111, projection='3d')


    x_dim, y_dim, z_dim = arr.shape
    x, y, z = np.meshgrid(np.arange(x_dim),
                          np.arange(y_dim),
                          np.arange(z_dim),
                          indexing='ij')

    x_coords = x[arr == 1]
    y_coords = y[arr == 1]
    z_coords = z[arr == 1]

    ax.scatter(x_coords, y_coords, z_coords)
    plt.xlabel('x')
    plt.ylabel('y')
    ax.set_zlabel('z')

    plt.figure(fig_num+1)

    fft_arr = fftshift(fftn(arr))
    mags = np.abs(fft_arr)

    center = len(arr[0,0,:]) // 2

    if center_slice:
        plt.imshow(arr[:, :, center])
    else:
        plt.imshow(arr[:, :, 0])

    plt.xlabel('x')
    plt.ylabel('y')

    plt.figure(fig_num+2)
    plt.imshow(mags[:, :, center])
    plt.xlabel('kx')
    plt.ylabel('ky')

    plt.figure(fig_num+3)
    if center_slice:
        plt.imshow(arr[:, center, :])
    else:
        plt.imshow(arr[:, 1, :])

    plt.xlabel('x')
    plt.ylabel('z')

    plt.figure(fig_num+4)
    plt.imshow(mags[:, center, :])
    plt.xlabel('kx')
    plt.ylabel('kz')

    plt.figure(fig_num+5)
    if center_slice:
        plt.imshow(arr[center, :, :])
    else:
        plt.imshow(arr[2, :, :])

    plt.imshow(arr[center, :, :])
    plt.xlabel('y')
    plt.ylabel('z')

    plt.figure(fig_num+6)
    plt.imshow(mags[:, center, :])
    plt.xlabel('ky')
    plt.ylabel('kz')

    fig = plt.figure(fig_num+7)
    ax = fig.add_subplot(111, projection='3d')

    kx_dim, ky_dim, kz_dim = fft_arr.shape

    kx, ky, kz = np.meshgrid(np.arange(kx_dim),
                          np.arange(ky_dim),
                          np.arange(kz_dim),
                          indexing='ij')

    kx_coords = kx[mags >= min_intensity]
    ky_coords = ky[mags >= min_intensity]
    kz_coords = kz[mags >= min_intensity]

    ax.scatter(kx_coords, ky_coords, kz_coords, c=mags[mags>=min_intensity])


def build_checkerboard(n):
    return np.array([[(i + j) % 2 for j in range(n)] for i in range(n)])


def build_BCC_lattice(shape, spacing):
    """Generate a 3D array with 1s at lattice points spaced by `spacing`."""
    lattice = np.zeros(shape)
    for x in range(0, shape[0], spacing):
        for y in range(0, shape[1], spacing):
            for z in range(0, shape[2], spacing):
                lattice[x, y, z] = 1
    return lattice


def tile_unit_cell(unit_cell_coords, tile_range=1):
    """ This function"""

    x_coords = unit_cell_coords[:, 0]
    y_coords = unit_cell_coords[:, 1]
    z_coords = unit_cell_coords[:, 2]

    x_coords_mod = x_coords + 100
    y_coords_mod = y_coords + 100
    z_coords_mod = z_coords + 100

    arrxmod = np.stack([x_coords_mod, y_coords, z_coords], axis=1)
    arrymod = np.stack([x_coords, y_coords_mod, z_coords], axis=1)
    arrzmod = np.stack([x_coords, y_coords, z_coords_mod], axis=1)
    arrxymod = np.stack([x_coords_mod, y_coords_mod, z_coords], axis=1)
    arrxzmod = np.stack([x_coords_mod, y_coords, z_coords_mod], axis=1)
    arryzmod = np.stack([x_coords, y_coords_mod, z_coords_mod], axis=1)
    arrxyzmod = np.stack([x_coords_mod, y_coords_mod, z_coords_mod], axis=1)

    eight_unit_cells = np.vstack((unit_cell_coords, arrxmod, arrymod, arrzmod, arrxymod, arrxzmod, arryzmod, arrxyzmod))

    return eight_unit_cells


def build_clathrate(x, y, z, tile_unit_cells=False):

    clath = Clathrate(x=x, y=y, z=z, lattice_parameter=10)
    coords = clath.pattern_3d('unit cell')

    clathrate_coords = np.vstack((coords[0], coords[1], coords[2], coords[3], coords[4]))

    clathrate_coords = np.round(clathrate_coords, 2)

    discrete_coords = (np.round(clathrate_coords*100)).astype(int)

    if tile_unit_cells:
        discrete_coords = tile_unit_cell(discrete_coords)

    max_coords = np.max(discrete_coords, axis=0) + 1
    space = np.zeros(tuple(max_coords), dtype=int)

    for x, y, z in discrete_coords:
        space[x, y, z] = 1

    np.save('../Data/Qlab3/eight_unit_cell_clathrate_array.npy', space)

    return space

if __name__ == '__main__':

    arr = build_clathrate(0.18, 0.31, 0.11, tile_unit_cells=True)

    print(f"There are {np.sum(arr)} atoms in this lattice")
    FFT_3D(arr,1, min_intensity=250)

    arr2 = build_BCC_lattice((501, 501, 501), 14)
    FFT_3D(arr2, 9, min_intensity=20000, center_slice=False)
    plt.show()

