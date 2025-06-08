import numpy as np
import math
import pandas as pd
import warnings
from itertools import groupby
from scipy.spatial import ConvexHull


def magnitude(*args: float) -> float:
    """returns the magnitude of a vector with n parameters"""
    number_of_params = len(args)
    magnitude_squared = 0
    for ii in range(number_of_params):

        if args[ii] is complex:
            raise TypeError("function does not support complex numbers")
        elif args[ii] == float('inf') or args[ii] == float('-inf'):
            raise ValueError("vector has one or more infinite parameters")

        magnitude_squared = magnitude_squared + args[ii] ** 2
    return math.sqrt(magnitude_squared)


class Clathrate:
    """Defines a clathrate and functions related to clathrates"""
    number_of_clathrates = 0

    """These are all of the atomic positions in the primitive unit cell that are fixed. They are held in an nx3 numpy 
    array"""

    a1 = np.array([0, 0, 0])
    a2 = np.array([0.5, 0.5, 0.5])
    wyckoff_2a = np.array([a1, a2])  # 2X3 array where each row is the 3D coordinate of a 2a atom

    c1 = np.array([0.25, 0, 0.5])
    c2 = np.array([0.75, 0, 0.5])
    c3 = np.array([0.5, 0.25, 0])
    c4 = np.array([0.5, 0.75, 0])
    c5 = np.array([0, 0.5, 0.25])
    c6 = np.array([0, 0.5, 0.75])
    wyckoff_6c = np.array([c1, c2, c3, c4, c5, c6])  # 6x3 array for 6c Wyckoff position

    d1 = np.array([0.25, 0.5, 0])
    d2 = np.array([0.75, 0.5, 0])
    d3 = np.array([0, 0.25, 0.5])
    d4 = np.array([0, 0.75, 0.5])
    d5 = np.array([0.5, 0, 0.25])
    d6 = np.array([0.5, 0, 0.75])
    wyckoff_6d = np.array([d1, d2, d3, d4, d5, d6])  # 6X3 array for 6d Wyckoff position

    """ Here is a dictionary of all the ionic radii with maximum coordination number"""
    ionic_radii = {'Na': 1.39, 'K': 1.64, 'Rb': 1.72, 'Cs': 1.88,
                   'Sr': 1.44, 'Ba': 1.61,
                   'Eu': 1.35, 'I': 2.20}
    crystal_radii = {'Zn': 0.74,
                     'B': 0.25, 'Al': 0.53, 'Ga': 0.61, 'In': 0.76,
                     'Si': 0.4, 'Ge': 0.53, 'Sn': 0.69,
                     'Sb': 0.9}
    guest_color = {'Na': 'red', 'K': 'green', 'Rb': 'blue', 'Cs': 'cyan', 'Sr': 'magenta', 'Ba': 'yellow',
                   'Eu': 'black', 'I': 'black'}

    # Rb XII=1.72, Rb XIV=1.83
    # Eu+2 X=1.35, Eu+3 IX=1.120
    # I-1 VI=2.20, I+5 VI=0.95, I+7 VI=0.53
    # CN (coordination number): Na=XII, K=XII, Rb=XII, Cs=XII, Sr=XII, Ba=XII, Eu=X, I=VI.
    """Error is stored in these dictionaries, if no error is input into the class, it is stored like this. Note that any
    error manually input must be in this format. The same goes for atomic displacement parameters and occupancies"""

    default_error = {'x': 0,
                     'y': 0,
                     'z': 0,
                     'lp': 0,
                     'adp': {'2a': 0, '6c': 0, '6d': 0, '16i': 0, '24k': 0},
                     'occ': {'2a': 0, '6c': 0, '6d': 0, '16i': 0, '24k': 0}
                     }

    default_atomic_displacement_parameters = {'2a': 0, '6c': 0, '6d': 0, '16i': 0, '24k': 0}

    default_occupancies = {'2a': 0, '6c': 0, '6d': 0, '16i': 0, '24k': 0}

    """Our init function inputs all of the variables into the class, and gives you warnings if you input x,y,z 
    coordinates that don't make sense physically"""
    def __init__(self, x: float, y: float, z: float, lattice_parameter=float('NaN'), occupancies=default_occupancies,
                 atomic_displacement_parameters=default_atomic_displacement_parameters, ionic_radius=float('NaN'),
                 color_value=float('NaN'),  guest_atom='?', name='Default Clathrate', framework_type='?',
                 error=default_error, rating='?', mono_cage = True):
        # Give warning if Wyckoff values don't make sense
        if x < 0 or y < 0 or z < 0:
            warnings.warn(f"\nWARNING: Wyckoff values cannot be negative! \nx={x:.3f}, y={y:.3f}, z={z:.3f}")
        elif x > 1/4:
            warnings.warn(f"\nWARNING: x should be <= 0.25! \nx={x:.3f}")
        elif y > 1/2:
            warnings.warn(f"\nWARNING: y must be <= 0.5! \ny={y:.3f}")
        elif z > 1/2:
            warnings.warn(f"\nWARNING: z should probably be < 0.5! \nz={z:.3f}")

        self.x = x
        self.y = y
        self.z = z
        self.lp = lattice_parameter  # in angstrom
        self.occ = occupancies
        self.adp = atomic_displacement_parameters
        self.ir = ionic_radius  # in angstrom
        self.cv = color_value  # based on ionic radius. This parameter helps to define color gradients later
        self.name = name  # defines the name of the clathrate for plotting purposes
        self.guest = guest_atom
        self.framework = framework_type
        self.error = error
        self.x_err = error['x']
        self.rating = rating
        self.mono_cage = mono_cage

        Clathrate.number_of_clathrates += 1

    """This function returns the clathrate's name when you type repr(clathrate)"""
    def __repr__(self) -> str:
        # return ("Clathrate(x={}, y={}, z={}, a={}, ir={}, cv={}, guest='{}', framework='{}', name='{}', "
        #         "occupancy = {}, atomic displacement parameter = {}, error = {}))"
        #         .format(self.x, self.y, self.z, self.lp, self.ir, self.cv, self.guest, self.framework, self.name,
        #                 self.occ, self.adp, self.error))

        return self.name

    """This function makes python label each instance of the clathrate class by its name in the excel file"""
    def __str__(self) -> str:
        return self.name

    """This function defines an nx3 numpy array for the 16i subgroup based on x"""
    def wyckoff_16i(self) -> np.ndarray:  # returns a 16X3 array where each row is the 3D coordinate of a 16i site atom
        i1 = np.array([self.x, self.x, self.x])
        i2 = np.array([-self.x, -self.x, self.x])
        i3 = np.array([-self.x, self.x, -self.x])
        i4 = np.array([self.x, -self.x, -self.x])
        i5 = np.array([self.x + 0.5, self.x + 0.5, -self.x + 0.5])
        i6 = np.array([-self.x + 0.5, -self.x + 0.5, -self.x + 0.5])
        i7 = np.array([self.x + 0.5, -self.x + 0.5, self.x + 0.5])
        i8 = np.array([-self.x + 0.5, self.x + 0.5, self.x + 0.5])
        i9 = np.array([-self.x, -self.x, -self.x])
        i10 = np.array([self.x, self.x, -self.x])
        i11 = np.array([self.x, -self.x, self.x])
        i12 = np.array([-self.x, self.x, self.x])
        i13 = np.array([-self.x + 0.5, -self.x + 0.5, self.x + 0.5])
        i14 = np.array([self.x + 0.5, self.x + 0.5, self.x + 0.5])
        i15 = np.array([-self.x + 0.5, self.x + 0.5, -self.x + 0.5])
        i16 = np.array([self.x + 0.5, -self.x + 0.5, -self.x + 0.5])
        return np.array([i1, i2, i3, i4, i5, i6, i7, i8, i9,
                        i10, i11, i12, i13, i14, i15, i16])  # 16X3 array for 16i Wyckoff position

    """This function defines an nx3 array for the 24k subgroup based on parameters y and z"""
    def wyckoff_24k(self) -> np.ndarray:
        k1 = np.array([0, self.y, self.z])
        k2 = np.array([0, -self.y, self.z])
        k3 = np.array([0, self.y, -self.z])
        k4 = np.array([0, -self.y, -self.z])
        k5 = np.array([self.z, 0, self.y])
        k6 = np.array([self.z, 0, -self.y])
        k7 = np.array([-self.z, 0, self.y])
        k8 = np.array([-self.z, 0, -self.y])
        k9 = np.array([self.y, self.z, 0])
        k10 = np.array([-self.y, self.z, 0])
        k11 = np.array([self.y, -self.z, 0])
        k12 = np.array([-self.y, -self.z, 0])
        k13 = np.array([self.y + 0.5, 0.5, -self.z + 0.5])
        k14 = np.array([-self.y + 0.5, 0.5, -self.z + 0.5])
        k15 = np.array([self.y + 0.5, 0.5, self.z + 0.5])
        k16 = np.array([-self.y + 0.5, 0.5, self.z + 0.5])
        k17 = np.array([0.5, self.z + 0.5, -self.y + 0.5])
        k18 = np.array([0.5, self.z + 0.5, self.y + 0.5])
        k19 = np.array([0.5, -self.z + 0.5, -self.y + 0.5])
        k20 = np.array([0.5, -self.z + 0.5, self.y + 0.5])
        k21 = np.array([self.z + 0.5, self.y + 0.5, 0.5])
        k22 = np.array([self.z + 0.5, -self.y + 0.5, 0.5])
        k23 = np.array([-self.z + 0.5, self.y + 0.5, 0.5])
        k24 = np.array([-self.z + 0.5, -self.y + 0.5, 0.5])
        return np.array([k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12,
                        k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23,
                        k24])

    """This function returns an array of all the possible bond magnitudes in the primitive unit cell between two 
    different ararys of atom positions"""
    def bond_length_array(self, wyckoff_array_long: np.ndarray, wyckoff_array_short: np.ndarray) \
            -> tuple[np.ndarray, np.ndarray]:

        if not hasattr(wyckoff_array_long, "__len__") or not hasattr(wyckoff_array_short, "__len__"):
            raise TypeError("one or more inputs are not arrays")

        if wyckoff_array_long.ndim == 1 and wyckoff_array_short.ndim == 1:
            bond_array = Clathrate.bond_vector(wyckoff_array_long, wyckoff_array_short)
            magnitude_array = magnitude(*bond_array)
        else:
            bond_array = Clathrate.bond_vector(wyckoff_array_long, wyckoff_array_short)
            [rows, _] = bond_array.shape
            magnitude_array = np.zeros(rows)
            for ii in range(rows):
                magnitude_array[ii] = magnitude(*bond_array[ii])
        return magnitude_array * self.lp, magnitude_array

    """This function returns the shortest possible bond distances between any 2 wyckoff subgroups"""
    def bond_length(self) -> dict:
        # Calculate bond length using the derived math functions. Letters are ordered alphabetically for cage bonds,
        # but with guest first for guest neighbor length.
        return {
            "6c-24k": math.sqrt((self.y - 1 / 2) ** 2 + (self.z - 1 / 4) ** 2),
            "16i-16i": math.sqrt(3 * (2 * self.x - 1 / 2) ** 2),
            "16i-24k": math.sqrt(self.x ** 2 + (self.x - self.y) ** 2 + (self.x - self.z) ** 2),
            "24k-24k": 2 * self.z,

            "2a-16i": math.sqrt(3) * self.x,
            "2a-24k": math.sqrt(self.y ** 2 + self.z ** 2),

            "6d-6c": 1 / math.sqrt(8),  # This is constant always
            "6d-16i": math.sqrt(3 * self.x * (self.x - 1 / 2) + 5 / 16),
            "6d-24k short": math.sqrt((self.y - 1 / 2) ** 2 + self.z ** 2 + 1 / 16),
            "6d-24k long": math.sqrt((self.y - 1 / 4) ** 2 + (self.z - 1 / 2) ** 2)
        }

    """This function returns the error for the bond length equations in the function above in the same format"""
    def bond_length_error(self) -> dict:
        # Output dictionary of error for bond lengths.
        error_x = self.error['x']
        error_y = self.error['y']
        error_z = self.error['z']

        ck_len = self.bond_length()["6c-24k"]
        ik_len = self.bond_length()["16i-24k"]
        ak_len = self.bond_length()["2a-24k"]
        di_len = self.bond_length()["6d-16i"]
        dk_short_len = self.bond_length()["6d-24k short"]
        dk_long_len = self.bond_length()["6d-24k long"]

        return {
            "6c-24k": 1/ck_len * (error_y * (1 / 2 - self.y) + error_z * (1 / 4 - self.z)),
            "16i-16i": math.sqrt(12) * error_x,
            "16i-24k": 1/ik_len * (error_x * abs(3 * self.x - self.y - self.z) + error_y * abs(self.y - self.x) +
                                   error_z * (self.x - self.z)),
            "24k-24k": 2 * error_z,

            "2a-16i": math.sqrt(3) * error_x,
            "2a-24k": 1/ak_len * (error_y * self.y + error_z * self.z),

            "6d-6c": 0,  # This is constant always
            "6d-16i": 1 / di_len * 3 * error_x * (1 / 4 - self.x),
            "6d-24k short": 1/dk_short_len * (error_y * (1 / 2 - self.y) + error_z * self.z),
            "6d-24k long": 1/dk_long_len * (error_y * abs(1 / 4 - self.y) + error_z * (1 / 2 - self.z))
        }

    """This function returns various volume parameters for both cages depending on input"""
    def volume_of_cage(self, which_cage, which_volume='effective volume', d_over_a_volume=True):

        a_arr, c_arr, d_arr, i_arr, k_arr = self.pattern_3d(which_cage)

        if 'large' in which_cage:
            # This is the volume of the 'twisted polyhedron' thing from the 24k points in the hexagons:
            effective_volume = 1/3 * (self.y**2 + self.z**2 - 4*self.y*self.z - self.y + 2*self.z + 1/4)
            points = np.concatenate((c_arr, d_arr, i_arr, k_arr))

        elif 'small' in which_cage:
            # This is the volume of the cube made by the 16i points:
            effective_volume = (2*self.x)**3
            points = np.concatenate((a_arr, i_arr, k_arr))

        else:
            return Exception("which_cage should be either 'large cage' or 'small cage'. ")

        hull = ConvexHull(points)
        actual_volume = hull.volume

        if d_over_a_volume is False:  # If you want the volume in terms of d, rather than d/a:
            effective_volume = effective_volume * self.lp**3
            actual_volume = actual_volume * self.lp**3

        if which_volume == 'effective volume':
            return effective_volume

        elif which_volume == 'actual volume':
            return actual_volume

    """This function defines the color of each wyckoff subgroup for plotting purposes"""
    def color_gradients(self) -> tuple:
        # red = (float(1), self.cv, self.cv)
        # orange = (float(1), 0.5 + self.cv / 2, self.cv / 2)
        # blue = (self.cv, self.cv, float(1))
        # purple = (0.5 + self.cv / 2, self.cv / 2, float(1))
        # green = (self.cv, float(.7), self.cv)
        #
        # return red, orange, blue, purple, green

        color1 = (0.3, 0.3, 0.3)
        color2 = (1, 0, 0)  # red
        color3 = (0.85, 0.2, 1)
        color4 = (0.02, 0.83, 0.07)  # green
        color5 = (0.19, 0.36, 1)  # blue

        return color1, color2, color3, color4, color5

    @staticmethod
    def bond_vector(wyckoff_array_long, wyckoff_array_short):
        """ returns an array of all possible bonds between 2 wyckoff sites
        For example, if you input 2a and 6c, the function will return all the possible bonds between 2a and 6d
        in the following order: [a1-d1, a1-d2, ..., a1-d6, a2-d1, ..., a2-d6] """

        # the first part of the function makes sure that the inputs are of the right type and value
        if not hasattr(wyckoff_array_long, "__len__") or not hasattr(wyckoff_array_short, "__len__"):
            raise TypeError("one or more inputs are not arrays")

        if wyckoff_array_long.ndim == 1 and wyckoff_array_short.ndim == 1:
            bond_vector_array = wyckoff_array_long - wyckoff_array_short

        else:
            bond_vector_array = np.zeros((wyckoff_array_long.shape[0] * wyckoff_array_short.shape[0], 3))
            for ii in range(wyckoff_array_long.shape[0]):
                for jj in range(wyckoff_array_short.shape[0]):
                    bond_vector_array[jj + ii * wyckoff_array_short.shape[0], [0, 1, 2]] = \
                        (wyckoff_array_long[ii] - wyckoff_array_short[jj])

        index = []
        for ii in range(len(bond_vector_array)):
            if np.array_equal(bond_vector_array[ii], np.array([0, 0, 0]), equal_nan=False):
                index.append(ii)
        bond_vector_array = np.delete(bond_vector_array, index, 0)
        return bond_vector_array

    @classmethod
    def all_clathrates_from_xlsx(cls, file_name: str) -> tuple:
        """ This method reads all clathrate data from ClathrateFormattedData.xlsx and sorts each clathrate by its
        framework type. It is kind of a messy function that I need to make more efficient, but it works!
        """
        df = pd.read_excel(file_name)
        df.set_index('Name', inplace=True)
        clathrates = []
        silicon_clathrates, boron_silicon_clathrates, aluminum_silicon_clathrates, gallium_silicon_clathrates = (
            [], [], [], [])
        germanium_clathrates, aluminum_germanium_clathrates = [], []
        tin_clathrates, gallium_tin_clathrates = [], []
        calculated_clathrates = []
        for name in df.index:
            if pd.isnull(df.loc[name, 'Rating (see notes, Column AC)']) is False:
                rating = df.loc[name, 'Rating (see notes, Column AC)']
            else:  # Will make sure rating is a string, rather than nan
                rating = '?'

            if 'ignore' in rating:
                continue  # Will skip this loop iteration

            x = df.loc[name, 'Wyckoff x']
            y = df.loc[name, 'Wyckoff y']
            z = df.loc[name, 'Wyckoff z']
            lp = df.loc[name, 'LP (Angstroms)']
            a_occ = df.loc[name, 'G1 (2a) occupancy']
            c_occ = df.loc[name, 'F1 (6c) occupancy (Framework 1, Framework 2)']
            d_occ = df.loc[name, 'G2 (6d/equiv) occupancy']
            i_occ = df.loc[name, 'F2 (16i) occupancy']
            k_occ = df.loc[name, 'F3 (24k) occupancy']
            a_ADP = df.loc[name, 'G1 (2a) atomic displacement parameter (ADP) [A^2]']
            c_ADP = df.loc[name, 'F1 (6c) ADP']
            d_ADP = df.loc[name, 'G2 (6d/24j) ADP']
            i_ADP = df.loc[name, 'F2 (16i) ADP']
            k_ADP = df.loc[name, 'F3 (24k) ADP']
            framework = df.loc[name, 'Framework 1 name']
            if pd.isnull(df.loc[name, 'Framework 2 name']) is False:
                mono_cage = False
                framework = f"{framework} {df.loc[name, 'Framework 2 name']}"
            else:
                mono_cage = True

            ''' takes error out of values and converts to float'''
            list_of_params = [x, y, z, lp, a_ADP, c_ADP, d_ADP, i_ADP, k_ADP, a_occ, c_occ, d_occ, i_occ, k_occ]
            list_of_err = []
            list_of_adp_val = []
            for adp in list_of_params:
                list_of_adp_val.append(cls.convertADPtoBeq(adp))

            for ii, (var, adp) in enumerate(zip(list_of_params, list_of_adp_val)):
                list_of_err.append(cls.get_error_from_xlsx(var, adp))

                if type(var) is str:
                    if var == '?':
                        param = float('NaN')

                    elif "," in var:
                        param = var.split(',')
                        param1 = param[0]
                        param2 = param[1]
                        param1 = float(param1.split('(')[0])
                        param2 = float(param2.split('(')[0])
                        param = [param1, param2]

                    else:
                        if "(" in var:
                            value = var.split('(')[0]

                        elif "U" in var:
                            value = var.split('U')[0]

                        elif "B" in var:
                            value = var.split("B")[0]

                        else:
                            value = var
                        param = float(value)

                        if 'U' in var:
                            param = param * 8 * (math.pi ** 2)

                        elif 'B' in var:
                            param = param

                        elif var[-1] == 'B':
                            param = param

                elif type(var) is not float and type(var) is not int:
                    raise TypeError("one or more datapoints are neither strings, floats, nor ints")

                else:
                    param = var
                list_of_params[ii] = param

            ADPs = {'2a': list_of_params[4], '6c': list_of_params[5], '6d': list_of_params[6], '16i': list_of_params[7],
                    '24k': list_of_params[8]}

            ADP_err = {'2a': list_of_err[4], '6c': list_of_err[5], '6d': list_of_err[6], '16i': list_of_err[7],
                       '24k': list_of_err[8]}

            occs = {'2a': list_of_params[9], '6c': list_of_params[10], '6d': list_of_params[11],
                    '16i': list_of_params[12], '24k': list_of_params[13]}

            occ_err = {'2a': list_of_err[9], '6c': list_of_err[10], '6d': list_of_err[11], '16i': list_of_err[12],
                       '24k': list_of_err[13]}

            error_dict = {'x': list_of_err[0], 'y': list_of_err[1], 'z': list_of_err[2], 'lp': list_of_err[3],
                          'adp': ADP_err, 'occ': occ_err}

            ir_key = df.loc[name, 'Guest 1 name']
            if type(ir_key) == str:
                ir_value = cls.ionic_radii[ir_key]

            elif 'C' in rating:  # If the clathrate entry is calculated (C), then basically just ignore it.
                ir_value = 999

            else:
                warnings.warn(f"\n\nWARNING, ir_key not str!"
                              f"\n   ir_key: {ir_key}\n   name: {name}\n   var: {var}\n")
                ir_value = 999

            cv = ir_value/2

            clathrate = cls(list_of_params[0], list_of_params[1], list_of_params[2], list_of_params[3], occs,
                            ADPs, ir_value, cv, ir_key, name, 'default framework', error_dict, rating,
                            mono_cage=mono_cage)

            if 'C' not in rating:  # Ignore calculated values
                clathrates.append(clathrate)
                clathrate.framework = framework

                if pd.isnull(df.loc[name, 'Guest 2 name']) is False:  # Ignore if clathrate has 2 guests
                    continue

                if 'Si' in framework:
                    if framework == 'Si': # If just Si, add to silicon group.
                        silicon_clathrates.append(clathrate)

                    elif 'Al' in framework:
                        aluminum_silicon_clathrates.append(clathrate)

                    elif 'Ga' in framework:
                        gallium_silicon_clathrates.append(clathrate)

                    elif 'B' in framework:
                        boron_silicon_clathrates.append(clathrate)

                elif 'Ge' in framework:
                    if framework == 'Ge':
                        germanium_clathrates.append(clathrate)
                        
                    elif 'Al' in framework:
                        aluminum_germanium_clathrates.append(clathrate)

                elif 'Sn' in framework:
                    if framework == 'Sn':
                        tin_clathrates.append(clathrate)

                    elif 'Ga' in framework:
                        gallium_tin_clathrates.append(clathrate)

            elif 'C' in rating:
                calculated_clathrates.append(clathrate)
                clathrate.framework = framework

        return (clathrates,
                silicon_clathrates, boron_silicon_clathrates, aluminum_silicon_clathrates, gallium_silicon_clathrates,
                germanium_clathrates, aluminum_germanium_clathrates,
                tin_clathrates, gallium_tin_clathrates,
                calculated_clathrates)

    @staticmethod
    def convertADPtoBeq(var: str):
        """this function takes an input from the ADP columns in data/ClathrateFormattedData.xlsx
        and converts it to Beq"""
        if type(var) is str:

            if "U" in var:
                adp_val = 'U'

            elif "B" in var:
                adp_val = 'B'
            else:
                adp_val = float('NaN')
        else:
            adp_val = float("NaN")

        return adp_val

    @staticmethod
    def get_error_from_xlsx(var, ADP_val):
        """returns error from input (e.g. var=3.0(3) returns 0.3 as the error"""
        if type(var) is str:
            if var == '?':
                error = 0

            elif "," in var:
                var = var.split(",")
                var1 = var[0]
                number1 = var1.split('(')[0]
                var2 = var[1]
                number2 = var2.split('(')[0]

                if "(" in var1:
                    error1 = var1.split('(')[1]
                    error1 = float(error1.split(')')[0])
                    digits_after_decimal = len(number1.split('.')[1])
                    error1 = error1 * 10**-digits_after_decimal

                else:
                    error1 = 0

                if "(" in var2:
                    error2 = var2.split('(')[1]
                    error2 = float(error2.split(')')[0])
                    digits_after_decimal = len(number2.split('.')[1])
                    error2 = error2 * 10**-digits_after_decimal

                else:
                    error2 = 0
                error = error1, error2

            else:
                if "(" in var:
                    var = var.split('(')
                    error = var[1]
                    error = float(error.split(')')[0])
                    digits_after_decimal = len(var[0].split('.')[1])
                    error = error * 10**-digits_after_decimal

                else:
                    error = 0
        elif type(var) is float or type(var) is int:
            error = 0

        else:
            raise ValueError(f"\nvar must be either string, float, or int. "
                             f"\n   Inputted var: {var}\n   ADP_val: {ADP_val}")
        
        if ADP_val == 'U':
            return error * 8 * (math.pi**2)

        else:
            return error

    @classmethod
    def from_xlsx_file(cls, name: str, file_name: str):
        """ returns a single clathrate from data/ClathrateFormattedData.xlsx"""
        clathrates = cls.all_clathrates_from_xlsx(file_name)[0] + cls.all_clathrates_from_xlsx(file_name)[-1]
        for clathrate in clathrates:
            if clathrate.name == name:
                return clathrate
        else:
            raise ValueError("name not found in data/ClathrateFormattedData.xlsx")

    def wyckoff_dict(self):
        """ returns a pandas dataframe containing information about each wyckoff site"""
        dictionary = {'name': ['2a', '6c', '6d', '16i', '24k'],
                      'position': [self.wyckoff_2a, self.wyckoff_6c, self.wyckoff_6d, self.wyckoff_16i(),
                                   self.wyckoff_24k()],
                      'multiplicity': [2, 6, 6, 16, 24],
                      'site': ['a', 'c', 'd', 'i', 'k'],
                      'color': ['red', 'orange', 'blue', 'purple', 'green']
                      }
        df = pd.DataFrame.from_dict(dictionary)
        return df

    def unit_cube_atom_positions(self, x_lim, y_lim, z_lim):
        """return array of atoms in clathrate unit cell"""
        wyckoff_2a_atoms = np.empty(3)
        wyckoff_6c_atoms = np.empty(3)
        wyckoff_6d_atoms = np.empty(3)
        wyckoff_16i_atoms = np.empty(3)
        wyckoff_24k_atoms = np.empty(3)
        operations = [np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]),
                      np.array([1, 1, 0]), np.array([1, 0, 1]), np.array([0, 1, 1]), np.array([1, 1, 1]),
                      np.array([-1, 0, 0]), np.array([0, -1, 0]), np.array([0, 0, -1])]

        dictionary = self.wyckoff_dict()
        for name, position, multiplicity in zip(dictionary['name'], dictionary['position'],
                                                dictionary['multiplicity']):

            for ii in range(multiplicity):
                for operation in operations:
                    new_position = (position[ii] + operation)
                    x, y, z = new_position
                    if x_lim[0] <= x <= x_lim[1] and y_lim[0] <= y <= y_lim[1] and z_lim[0] <= z <= z_lim[1]:
                        if name == '2a':
                            wyckoff_2a_atoms = np.vstack((wyckoff_2a_atoms, new_position))
                        elif name == '6c':
                            wyckoff_6c_atoms = np.vstack((wyckoff_6c_atoms, new_position))
                        elif name == '6d':
                            wyckoff_6d_atoms = np.vstack((wyckoff_6d_atoms, new_position))
                        elif name == '16i':
                            wyckoff_16i_atoms = np.vstack((wyckoff_16i_atoms, new_position))
                        elif name == '24k':
                            wyckoff_24k_atoms = np.vstack((wyckoff_24k_atoms, new_position))

        unit_cube_atoms = [wyckoff_2a_atoms, wyckoff_6c_atoms,wyckoff_6d_atoms, wyckoff_16i_atoms, wyckoff_24k_atoms]

        for ii, wyckoff_atoms in enumerate(unit_cube_atoms):
            unit_cube_atoms[ii] = np.delete(wyckoff_atoms, 0, axis=0)

        return unit_cube_atoms

    def pattern_3d(self, pattern):
        """Outputs the xyz coordinates for all the atoms in the chosen pattern.
        If an array will be empty, set it as an empty list. This way the later code knows it's empty."""

        if pattern == 'large cage':
            lim = [-999, 999]
            all_wyckoff_sites = self.unit_cube_atom_positions(lim, lim, lim)
            aCoordinates = np.array([[]])
            cCoordinates = (all_wyckoff_sites[1])[[0, 11, 22, 42], :]
            dCoordinates = np.array([(all_wyckoff_sites[2])[44]])
            iCoordinates = (all_wyckoff_sites[3])[[0, 12, 53, 55, 110, 122, 163, 165], :]
            kCoordinates = (all_wyckoff_sites[4])[[44, 67, 88, 100, 110, 122, 185, 198, 229, 231, 251, 253], :]

        elif pattern == 'small cage':
            lim = [-999, 999]
            all_wyckoff_sites = self.unit_cube_atom_positions(lim, lim, lim)
            aCoordinates = np.array([(all_wyckoff_sites[0])[0]])
            cCoordinates = np.array([[]])
            dCoordinates = np.array([[]])
            iCoordinates = (all_wyckoff_sites[3])[[0, 11, 22, 33, 88, 99, 110, 121], :]
            kCoordinates = (all_wyckoff_sites[4])[[0, 11, 22, 33, 44, 55, 66, 77, 88, 99, 110, 121], :]

        elif pattern == 'unit cell':
            lim = [0, 1]
            all_wyckoff_sites = self.unit_cube_atom_positions(lim, lim, lim)
            aCoordinates = all_wyckoff_sites[0]
            cCoordinates = all_wyckoff_sites[1]
            dCoordinates = all_wyckoff_sites[2]
            iCoordinates = all_wyckoff_sites[3]
            kCoordinates = all_wyckoff_sites[4]

        elif pattern == 'both cages':
            lim = [-999, 999]
            all_wyckoff_sites = self.unit_cube_atom_positions(lim, lim, lim)
            aCoordinates = np.array([(all_wyckoff_sites[0])[0]])
            cCoordinates = (all_wyckoff_sites[1])[[0, 11, 22, 42], :]
            dCoordinates = np.array([(all_wyckoff_sites[2])[44]])
            iCoordinates = (all_wyckoff_sites[3])[[0, 12, 53, 55, 110, 122, 163, 165, 11, 22, 33, 88, 99, 110, 121], :]
            kCoordinates = (all_wyckoff_sites[4])[
                           [0, 11, 22, 33, 44, 55, 66, 77, 88, 99, 110, 121, 44, 67, 88, 100, 110, 122, 185, 198, 229,
                            231, 251, 253], :]

        elif pattern == '6c neighbors':
            aCoordinates = np.array([[]])
            cCoordinates = np.array([[0, 0, 0]])
            dCoordinates = np.array([[0, 0.25, -0.25], [0, -0.25, -0.25], [-0.25, 0, 0.25], [0.25, 0, 0.25]])
            iCoordinates = np.array([[]])
            kCoordinates = np.array([[self.y-0.5, 0, self.z-0.25], [0.5-self.y, 0, self.z-0.25],
                                     [0, 0.5-self.y, 0.25-self.z], [0, self.y-0.5, 0.25-self.z]])

        elif pattern == '16i neighbors':
            aCoordinates = np.array([[-self.x, -self.x, -self.x]])
            cCoordinates = np.array([[]])
            dCoordinates = np.array([[0.25-self.x, 0.5-self.x, -self.x], [-self.x, 0.25-self.x, 0.5-self.x],
                                     [0.5-self.x, -self.x, 0.25-self.x]])
            iCoordinates = np.array([[0, 0, 0], [0.5-2*self.x, 0.5-2*self.x, 0.5-2*self.x]])
            kCoordinates = np.array([[-self.x, self.y-self.x, self.z-self.x], [self.y-self.x, self.z-self.x, -self.x],
                                     [self.z-self.x, -self.x, self.y-self.x]])

        elif pattern == '24k neighbors':
            aCoordinates = np.array([[0, -self.y, -self.z]])
            cCoordinates = np.array([[0, 0.5-self.y, 0.25-self.z]])
            dCoordinates = np.array([[-0.25, 0.5-self.y, -self.z], [0.25, 0.5-self.y, -self.z],
                                     [0, 0.25-self.y, 0.5-self.z]])
            iCoordinates = np.array([[-self.x, self.x-self.y, self.x-self.z], [self.x, self.x-self.y, self.x-self.z]])
            kCoordinates = np.array([[0, 0, 0], [0, 0, -2*self.z]])

        else:  # Primitive unit cell (all Wyckoff values)
            if pattern != '':
                print('Pattern not recognized, defaulting to primitive unit cell')
            aCoordinates = self.wyckoff_2a
            cCoordinates = self.wyckoff_6c
            dCoordinates = self.wyckoff_6d
            iCoordinates = self.wyckoff_16i()
            kCoordinates = self.wyckoff_24k()

        return aCoordinates, cCoordinates, dCoordinates, iCoordinates, kCoordinates


"""This function takes a list of clathrates and if there are any repeats in composition, it only takes the clathrate 
with the greatest occupancy"""
def get_only_clathrates_with_max_occupancy(clathrates: list):
    """This function takes a list of clathrates and if there are two or more instances of the same clathrates with
    different occupancies, it only adds with clathrate with the higher occupancy to the list"""

    keyfunc = lambda x: x.guest
    groups = []
    uniquekeys = []
    clathrates = sorted(clathrates, key=keyfunc)

    for k, g in groupby(clathrates, keyfunc):
        groups.append(list(g))  # Store group iterator as a list
        uniquekeys.append(k)

    new_clathrates = []
    for group in groups:
        occupancies = []

        for clathrate in group:

            if hasattr(clathrate.occ['2a'], '__len__'):
                occ_2a = sum(clathrate.occ['2a'])

            else:
                occ_2a = clathrate.occ['2a']

            if hasattr(clathrate.occ['6d'], '__len__'):
                occ_6d = sum(clathrate.occ['6d'])

            else:
                occ_6d = clathrate.occ['6d']

            occupancies.append(occ_2a + occ_6d)

        for ii, occupancy in enumerate(occupancies):

            if pd.isnull(occupancy):
                occupancies[ii] = 0

        ind = occupancies.index(max(occupancies))
        new_clathrates.append(group[ind])

    return new_clathrates
