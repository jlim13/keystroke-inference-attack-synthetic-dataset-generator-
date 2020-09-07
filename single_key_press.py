import pygame, sys
import random
import os
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
import skimage
import matplotlib.pyplot as plt
# import crop_keys
from skimage.filters import gaussian
import argparse
# key press dictionary for users only using their right hands

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def blurSurf(surface, amt):
    """
    Blur the given surface by the given 'amount'.  Only values 1 and greater
    are valid.  Value 1 = no blur.
    """
    if amt < 1.0:
        raise ValueError("Arg 'amt' must be greater than 1.0, passed in value is %s"%amt)
    scale = 1.0/float(amt)
    surf_size = surface.get_size()
    scale_size = (int(surf_size[0]*scale), int(surf_size[1]*scale))
    surf = pygame.transform.smoothscale(surface, scale_size)
    surf = pygame.transform.smoothscale(surf, surf_size)
    return surf

def gen_key(key_press_dict):
    #get random index

    r_idx = random.randint(0,len(key_press_dict)-1)

    key, coords = list(key_press_dict.items())[r_idx]

    if key == 'space':
        rn_x = random.uniform(-45.0, 45.0)
    elif key == 'backspace':
        rn_x = random.uniform(-6.0, 6.0)
    elif key == 'uppercase':
        rn_x = random.uniform(-6.0, 6.0)
    else:
        rn_x = random.uniform(-3.0, 3.0)

    rn_y = random.uniform(-5.0, 5.0)
    noise_coords = [list(coords)[0] + rn_x, list(coords)[1]+rn_y]

    return key, noise_coords

def create_blurry_phone(blurred_image):


    blurred_image = (blurred_image * 255 / np.max(blurred_image)).astype('uint8')

    blurred_image = Image.fromarray(blurred_image)

    to_darken = ImageEnhance.Brightness(blurred_image)
    dark_level = random.uniform(0.05,.55)
    blurred_image = to_darken.enhance(dark_level)
    blurred_image.save('xr_blur.png')
    phoneImage = pygame.image.load("xr_blur.png").convert() #change to xr_blur.png

    return phoneImage, blurred_image

def create_thumb(key_press_dict, blurred_image, left_handed):


    #let's create a surface to hold our ellipse:
    surface = pygame.Surface((230, 90),pygame.SRCALPHA)

    thumb_colors = [
        (105, 95, 85),
        (60, 46, 40),
        (75,57,50),
        (90,69,60),
        (105,80,70),
        (120,92,80),
        (135,103,90),
        (150,114,100),
        (165,126,110),
        (180,138,120),
        (195,149,130),
        (210,161,140),
        (225,172,150),
        (240,184,160),
        (255,195,170),
        (255,206,180)
    ]

    thumb_color = random.choice(thumb_colors)


    size = (0, 0, 230, 90)



    ellipse = pygame.draw.ellipse(surface, thumb_color, size)

    #new surface variable for clarity (could use our existing though)

    rand_noise = random.uniform(-5.0, 5.0)
    noisy_thumb = -75.  + rand_noise

    surface2 = pygame.transform.rotate(surface, noisy_thumb)

    ###### Create a left thumb
    if left_handed:
        surface2 = pygame.transform.flip(surface2, True, False)

    ###### Blur the thumb
    blur_level = random.uniform(1.0, 10.0)
    surface2 = blurSurf(surface2, blur_level)

    return surface2, thumb_color

def gen_random_block():
    surface = pygame.Surface((100, 100),pygame.SRCALPHA)

    colors = [
        (105, 95, 85),
        (60, 46, 40),
        (75,57,50),
        (90,69,60),
        (105,80,70),
        (120,92,80),
        (135,103,90),
        (150,114,100),
        (165,126,110),
        (180,138,120),
        (195,149,130),
        (210,161,140),
        (225,172,150),
        (240,184,160),
        (255,195,170),
        (255,206,180)
    ]

    color = random.choice(colors)
    size_x = random.randint(5,35)
    size_y = random.randint(5,35)
    size = (0, 0, size_x, size_y)

    ellipse = pygame.draw.ellipse(surface, color, size)

    return surface

def center_loc(x_min, x_max, y_min, y_max):
    center_x = x_min + ((x_max - x_min)/2)
    center_y = y_min + ((y_max - y_min)/2)
    return center_x, center_y


def gen_thumb(key_press_dict,
                xr_im,
                data_path,
                max_frames,
                save,
                gaussian_var = None):

    screen = pygame.display.set_mode((312, 632))
    running = True


    count = 0
    while running:


        im = Image.open(xr_im)
        #add semantic labels here


        #blurred_image = im.filter(ImageFilter.GaussianBlur(radius = blur_level))
        blurred_image = np.array(im, dtype = np.float32)
        #cam_noises = ['../data/0_noise.png.npy', '../data/1_noise.png.npy']
        #this_noise = random.choice(cam_noises)
        #this_noise = np.load(this_noise)
        #noise_im = Image.fromarray(this_noise, 'RGB')

        phone_im = blurred_image[:,:,:3]
        phone_im /= 255.

        '''below for additive noise'''

        # screen_only_mask = np.load('xr_screen_only_mask.npy')
        # to_morph = screen_only_mask == 1
        # untouched = screen_only_mask == 0
        #
        # original_pix_coords = np.where(screen_only_mask == 0)
        # original_pix_values = phone_im[untouched]
        # blurred_image = skimage.util.random_noise(phone_im,
        #                                         mode = noise_type,
        #                                         var = gaussian_var) #speckle
        #
        # #blurred_image = np.asarray(blurred_image * 255.0, dtype = np.uint8)
        # blurred_image[original_pix_coords[0], original_pix_coords[1]] = original_pix_values
        # # phone_im[zip(original_pix_coords)] = original_pix_values
        #
        # phone_im = blurred_image
        # blurr_to_save = Image.fromarray((phone_im * 255).astype(np.uint8), 'RGB')
        # #blurr_to_save.save("single_key_gaussian_{0}.png".format(gaussian_var))

        '''above, additive noise'''

        blurred_image = gaussian(phone_im, sigma=3, multichannel= True)


        left_handed = bool(random.getrandbits(1))

        thumb_surface, thumb_color = create_thumb(key_press_dict,
                                    blurred_image,
                                    left_handed)

        screen.fill((225 ,225, 215))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()


        phoneImage, _ = create_blurry_phone(blurred_image)

        screen.blit(phoneImage, (0,0))

        key, noise_coords = gen_key(key_press_dict)

        if left_handed:
            noise_coords[0] -= 80
        key_press_dir =  data_path + key + '/'

        if not os.path.exists(key_press_dir):
            os.makedirs(key_press_dir)


        _, _, files = next(os.walk(key_press_dir))
        file_count = len(files)

        key_press_dir =  key_press_dir + '_' + str(file_count) + '.png'

        screen.blit(thumb_surface, noise_coords)
        pygame.display.update()

        if save:
            pygame.image.save(screen, key_press_dir)


        count+=1
        if count == max_frames:

            running = False
            break


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    key_press_dict = {
        'a' : (15, 460),
        'b' : (160, 500),
        'c' : (100, 500),
        'd' : (75, 460),
        'e' : (50,420),
        'f' : (105, 460),
        'g' : (135, 460),
        'h' : (155, 460),
        'i' : (200, 420),
        'j' : (180,460),
        'k' : (210,460),
        'l' : (240, 460),
        'm' : (210, 500),
        'n' : (185, 500),
        'o' : (230, 420),
        'p' : (260, 420),
        'q' : (0,420),
        'r' : (80,420),
        's' : (45, 460),
        't' : (110, 420),
        'u' : (170, 420),
        'v' : (130, 500),
        'w' : (20,420) ,
        'x' : (75, 500),
        'y' : (140, 420),
        'z' : (50, 500),
        'space' : (130, 530),
        'enter' : (230, 530),
        'uppercase' : (0, 500),
        'digits' : (25, 530),
        'backspace' : (250, 500)
    }

    parser.add_argument("--left_handed", type = str2bool, default = False )
    parser.add_argument("--save", type = str2bool, default = True )
    parser.add_argument("--xr_im", type = str, default = 'xr_.png')
    parser.add_argument("--max_frames", type = int, default = 10000)
    parser.add_argument("--data_path", type = str, default =  '../data/single_key/')

    args = parser.parse_args()

    gen_thumb(key_press_dict,
                args.xr_im,
                args.data_path,
                args.max_frames,
                args.save,
                gaussian_var = None)
