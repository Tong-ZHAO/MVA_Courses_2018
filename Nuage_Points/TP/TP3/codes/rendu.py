from PIL import Image
import math
import numpy as np

def read_image(file_path):
    """Read a normal image from 4-channel png file and return its normalized normal

    Params:
        file_path (str): the path to the image

    Returns:
        im_normal (np.array): the centralized normal, it is normalized to [-1, 1].
    """

    im = np.array(Image.open(file_path)).astype(float)
    im_normal = im[:, :, :3]

    # Centralization
    im_normal_mean = im_normal.mean(axis = (0, 1), keepdims = True)
    im_normal -= im_normal_mean

    # Normalization
    im_normal_max = np.abs(im_normal).max()

    return im_normal / im_normal_max


def three_dim_dot(mat1, mat2):
    """Perform an element-wise vector dot on the third channel.
    
    1. The dimension of two matrix should be the same
    2. The result is clipped to [0, 1]

    Params:
        mat1 (np.array): m x n x k
        mat2 (np.array): m x n x k

    Returns:
        mat_dot (np.array): m x n x 1
    """

    assert(mat1.shape == mat2.shape), "The two matrices should have the same shape!"

    return np.clip((mat1 * mat2).sum(-1)[:, :, np.newaxis], 0, 1)


class Material_Lambert:
    """Lambert BRDF

    Attribus:
        my_color(np.array): the albedo color of size 1 x 1 x 3
        my_diffu(float): the diffusion coeff
    """

    def __init__(self, my_color, my_diffu):

        self.color = np.array(my_color).reshape((1, 1, -1))
        self.my_diffu = my_diffu

    def calc_fd(self):

        f =  self.my_diffu * self.color / np.pi

        return f


class Material_BP:
    """Blinn-Phong BRDF

    Attribus:
        coeff_specular (float): k^s
        coeff_shininess (float): S
    """

    def __init__(self, coeff_specular, coeff_shininess):

        self.coeff_specular = coeff_specular
        self.coeff_shininess = coeff_shininess

    def calc_fs(self, normal, w_i, w_o):

        w_h = w_i + w_o
        w_h /= np.linalg.norm(w_h, axis = -1, keepdims = True)

        f = self.coeff_specular * np.float_power(three_dim_dot(normal, w_h), self.coeff_shininess)

        return f


class Material_CT:
    """Cook-Torrance BRDF

    Attribus:
        alpha (float): roughness
        beta  (binary): metallic property, 1 - albedo, 0 - fresnel
        specular (np.array): albedo color, 1 x 1 x 3
        index (float): fresnel index
    """

    def __init__(self, alpha, beta, specular, index):

        self.alpha = alpha
        self.beta = beta
        self.specular = np.array(specular).reshape((1, 1, -1))
        self.index = index

    def calc_d(self, normal, w_h):

        dot_n_wh = three_dim_dot(normal, w_h)
        denom = np.pi * np.power((self.alpha ** 2 - 1) * np.power(dot_n_wh, 2) + 1, 2)

        return np.power(self.alpha, 2) / denom 

    def calc_f(self, w_i, w_h):

        c = np.power((self.index - 1) / (self.index + 1), 2)
        c = np.array([c, c, c]).reshape((1, 1, -1)) if self.beta == 0 else self.specular

        dot_wi_wh = three_dim_dot(w_i, w_h)

        return c + (1 - c) * np.power(1 - dot_wi_wh, 5)


    def calc_g(self, normal, w_i, w_o):

        k = self.alpha * np.sqrt(2 / np.pi)

        dot_n_wi = three_dim_dot(normal, w_i)
        dot_n_wo = three_dim_dot(normal, w_o)

        g1 = lambda my_dot: my_dot / (my_dot * (1 - k) + k)
        g_wi = g1(dot_n_wi)
        g_wo = g1(dot_n_wo)

        return g_wi * g_wo


    def calc_fs(self, normal, w_i, w_o):

        w_h = w_i + w_o
        w_h /= np.linalg.norm(w_h, axis = -1, keepdims = True)

        my_d = self.calc_d(normal, w_h)
        my_f = self.calc_f(w_i, w_h)
        my_g = self.calc_g(normal, w_i, w_o)

        dot_n_wi = three_dim_dot(normal, w_i)
        dot_n_wo = three_dim_dot(normal, w_o)

        denom = 4 * dot_n_wi * dot_n_wo
        denom[denom == 0] = 1.

        f =  my_d * my_f * my_g / denom

        return f


class LightSource:
    """Define light source

    Attribus:
        coord (np.array): the coordinate of the light source, 1 x 1 x 3
        color (np.array): albedo color, 1 x 1 x 3
        indense (float): the indensity of the light
    """

    def __init__(self, my_coord, my_color, my_ind):

        self.coord = np.array(my_coord).reshape((1, 1, -1))
        self.color = np.array(my_color).reshape((1, 1, -1))
        self.indense = my_ind

    def calc_wi(self, img_coords):

        w_i = img_coords - self.coord
        w_i /= np.linalg.norm(w_i, axis = -1, keepdims = True)

        return w_i

    def calc_li(self):

        return self.color * self.indense


class CameraDirection:
    """Define the camera direction

    Attribus:
        coord (np.array): the coordinate of the camera, 1 x 1 x 3
    """

    def __init__(self, my_coord):

        self.coord = np.array(my_coord).reshape((1, 1, -1))

    def calc_wo(self, img_coords):

        w_o = img_coords - self.coord
        w_o /= np.linalg.norm(w_o, axis = -1, keepdims = True) 

        return w_o


def shade(normalImage, method = "l"):
    """ Shade a normal image with different BRDF

    Params:
        normalImage (np.array): n x m x 3
        method (str): 'l' - Diffuse: lambert, Specular: 0
                      'bp' - Diffuse: lambert, Specular: Blinn-Phong
                      'ct' - Diffuse: lambert, Specular: Cook-Torrance

    Returns:
        Lo (np.array): n x m x 3, normalized output image (scaled to [0, 1])
    """

    method = method.lower()
    assert(method in ["l", "bp", "ct"]), "The BRDF is not implemented!"

    # defind light and BRDF model
    light = LightSource([0, 1., 1.], [0.5, 0.6, 0.6], 1)
    fd_brdf, fs_brdf = Material_Lambert([0.6, 0.75, 0.5], 1), None
    
    if method == 'bp':
        fs_brdf = Material_BP(0.5, 5)
    elif method == 'ct':
        fs_brdf = Material_CT(0.5, 0, [0.6, 0.75, 0.5], 4) 
    
    camera = CameraDirection([50., 400., 1.])
    h, w, _ = normalImage.shape

    # expand 2D image to 3D point cloud
    img_xcoords, img_ycoords = np.mgrid[0:h, 0:w]
    img_coords = np.stack((img_xcoords, img_ycoords, np.zeros_like(img_xcoords)), axis = -1)
    
    # calculate relating variables
    n_i = normalImage
    w_i = light.calc_wi(img_coords)
    w_o = camera.calc_wo(img_coords)

    f_d = fd_brdf.calc_fd()
    f_s = np.array([0]) if method == 'l' else fs_brdf.calc_fs(n_i, w_i, w_o)

    Lo = light.calc_li() * (f_s + f_d) * three_dim_dot(w_i, n_i)

    # normalize Lo to [0, 1]
    Lo_min, Lo_max = Lo.min(), Lo.max()
    Lo = (Lo - Lo_min) / (Lo_max - Lo_min)

    return Lo


if __name__ == "__main__":

    im = read_image("../normal.png")

    # Lambert
    shade_im = shade(im, 'l')
    shade_im = np.stack((shade_im))
    result = Image.fromarray((shade_im * 255).astype(np.uint8))
    result.show(title = "Lambert")

    # Blinn-Phong
    shade_im = shade(im, 'bp')
    shade_im = np.stack((shade_im))
    result = Image.fromarray((shade_im * 255).astype(np.uint8))
    result.show(title = "Lambert + Blinn-Phong")
    

    # Cook-Torrance
    shade_im = shade(im, 'ct')
    shade_im = np.stack((shade_im))
    result = Image.fromarray((shade_im * 255).astype(np.uint8))
    result.show(title = "Lambert + Cook-Torrance")
