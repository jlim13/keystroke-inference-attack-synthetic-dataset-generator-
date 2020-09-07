import pygame, sys
import random
import os
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
import skimage
import matplotlib.pyplot as plt
# import crop_keys
from skimage.filters import gaussian

blur_level = 4
# key press dictionary for users only using their right hands

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

    # screen_only_mask = np.load('xr_screen_only_mask.npy')
    # to_morph = screen_only_mask == 1
    # untouched = screen_only_mask == 0
    #
    # original_pix_coords = np.where(screen_only_mask == 0)
    # original_pix_values = blurred_image[untouched]


    # #add noise only to the screen
    # blurred_image = np.asarray(blurred_image, dtype = np.float64)
    # #print (np.unique(blurred_image))
    # blurred_image *= 1.0/blurred_image.max()
    # #print (np.unique(blurred_image))
    # blurred_image = skimage.util.random_noise(blurred_image, mode = noise_type) #speckle
    #
    # blurred_image = np.asarray(blurred_image * 255.0, dtype = np.uint8)
    # blurred_image[zip(original_pix_coords)] = original_pix_values
    blurred_image = (blurred_image * 255 / np.max(blurred_image)).astype('uint8')

    blurred_image = Image.fromarray(blurred_image)
    #blurred_image.save('just_noise.png')

    to_darken = ImageEnhance.Brightness(blurred_image)
    dark_level = random.uniform(0.05,.55)
    blurred_image = to_darken.enhance(dark_level)
    blurred_image.save('xr_blur.png')
    phoneImage = pygame.image.load("xr_blur.png").convert() #change to xr_blur.png

    return phoneImage, blurred_image

def create_thumb(key_press_dict, blurred_image, add_noise, left_handed, add_blit, noise_type ):


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

    #if we want to add some noise to the thumb
    if add_blit:
        rand_blit_x = random.randint(180,295)
        rand_blit_y = random.randint(75,150)
        blit_color = random.choice(thumb_colors)
        pygame.draw.ellipse(surface, blit_color, (0,0,rand_blit_x, rand_blit_y))

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

def segment_out_keys(xr_im, key_press_dict):

    im = plt.imread(xr_im)
    new_im = im.copy()

    all_centers = []

    start = 17

    bboxes = {}

    for key in ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p']:
        #letters.append(key)
        y_min = 419
        y_max = 451
        x_min = start
        x_max = start+24

        center = center_loc (x_min, x_max, y_min, y_max)
        all_centers.append(center)
        key_crop = im[y_min:y_max, x_min:x_max, :]
        bboxes[key] = [key_crop, [x_min, x_max, y_min, y_max], key]
        start += 28

    start = 31
    for key in ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l']:
        y_min = 457
        y_max = 489
        x_min = start
        x_max = start+24

        center = center_loc (x_min, x_max, y_min, y_max)
        all_centers.append(center)

        key_crop = im[y_min:y_max, x_min:x_max, :]
        bboxes[key] = [key_crop, [x_min, x_max, y_min, y_max], key]
        start += 28

    start = 59
    for key in ['z', 'x', 'c', 'v', 'b', 'n', 'm']:
        y_min = 495
        y_max = 527
        x_min = start
        x_max = start+24

        center = center_loc (x_min, x_max, y_min, y_max)
        all_centers.append(center)

        key_crop = im[y_min:y_max, x_min:x_max, :]
        bboxes[key] = [key_crop, [x_min, x_max, y_min, y_max], key]
        start += 28


    xs = [x[0] for x in all_centers]
    ys = [x[1] for x in all_centers]

    return bboxes, all_centers

def add_semantic_labels(xr_im, key_press_dict):

    def _color_mapping(num_keys):
        cmap = {}
        for key in num_keys:
            c = np.random.rand(3,) * 255.
            c = np.asarray(c, dtype = np.uint8)
            cmap[key] = c

        return cmap

    key_boxes, all_centers = segment_out_keys(xr_im, key_press_dict)
    num_semantic_labels = len(list(key_boxes.keys())) + 4 # screen, thumb, background, general keyboard

    label_dict = {}
    # label_dict['screen'] = 2
    # label_dict['thumb'] = 1
    # label_dict['background'] = 0
    im = plt.imread(xr_im)
    h,w,c = im.shape
    label_im = np.zeros((h,w,num_semantic_labels), dtype = np.uint8)
    label_im_debug_color = np.zeros((h,w,3), dtype = np.uint8)
    cmap = _color_mapping(list(key_boxes.keys()))


    label_im_debug_color[0:375, :, :] = np.array((233, 80, 10)) # screen
    label_im_debug_color[375:h, :, :] = np.array((80, 200, 200)) #keyboard

    for label, (k,v) in enumerate(key_boxes.items()):
        crop, bbox, letter = v
        y_min, y_max, x_min, x_max = bbox
        label_im[bbox, label] = label
        label_im[bbox, (len(list(key_boxes.keys()))+1)] = (len(list(key_boxes.keys()))+1)
        label_im_debug_color[x_min:x_max, y_min:y_max]= cmap[letter]




    plt.imshow(label_im_debug_color)
    plt.show()

    print (label_im)
    exit()

    return bboxes

def gen_thumb(orig_press_dict,
                xr_im,
                add_noise,
                add_blit,
                noise_type,
                data_path,
                random_keyboard,
                max_frames,
                save,
                gaussian_var = None):

    screen = pygame.display.set_mode((312, 632))
    running = True


    count = 0
    while running:
        if random_keyboard:
            xr_im, key_press_dict, _ = crop_keys.randomize_keyboard(xr_im, orig_press_dict)

        else:
            key_press_dict = orig_press_dict

        im = Image.open(xr_im)
        #add semantic labels here

        #label_im = add_semantic_labels(xr_im, key_press_dict)

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
                                    add_noise,
                                    left_handed,
                                    add_blit,
                                    noise_type)

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
        no_thumb_dir = '../data/single_key_no_thumb/'
        if not os.path.exists(no_thumb_dir):
            os.makedirs(no_thumb_dir)
        no_thumb_dir = no_thumb_dir + key + '/'
        if not os.path.exists(no_thumb_dir):
            os.makedirs(no_thumb_dir)
        if not os.path.exists(key_press_dir):
            os.makedirs(key_press_dir)


        path, dirs, files = next(os.walk(key_press_dir))
        file_count = len(files)

        key_press_dir =  key_press_dir + '_' + str(file_count) + '.png'

        no_thumb_dir = no_thumb_dir + '_' + str(file_count) + '.png'
        # if save:
        #     pygame.image.save(screen, no_thumb_dir)

        screen.blit(thumb_surface, noise_coords)
        print (screen.get_at((3, 4)))
        print (screen)
        exit(0)

        # for n in range(random.randint(5,10)):
        #     block = gen_random_block()
        #     block_x = random.randint(50,250)
        #     block_y = random.randint(50,550)
        #     screen.blit(block, [block_x, block_y] )

        pygame.display.update()

        if save:
            pygame.image.save(screen, key_press_dir)
            #pygame.image.save(screen, 'small_blocks.png')

        count+=1
        if count == max_frames:

            running = False
            break


if __name__ == "__main__":

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
        'uppercase' : (0, 500) ,
        'digits' : (25, 530),
        'backspace' : (250, 500)
    }

    add_noise = True
    left_handed = True
    add_blit = False
    noise_type = 'gaussian'
    random_keyboard = False
    save = True
    xr_im = 'xr_.png'
    max_frames = 10000 #same number of real life
    # xr_im, key_press_dict = crop_keys.randomize_keyboard(xr_im, key_press_dict)

    data_path = '../data/single_key/'
    # gaussian_vars = [.001, .01, .02, .05, .1, .15]
    # data_paths = ['../data/single_key_gaussian_std_{0}/'.format(x) for x in gaussian_vars]
    # for data_path, gaussian_var in zip(data_paths, gaussian_vars):
    #     gaussian_var = .75
    #     data_path = '../data/single_key_gaussian_std_.75/'
    gen_thumb(key_press_dict,
                xr_im,
                add_noise,
                add_blit,
                noise_type,
                data_path,
                random_keyboard,
                max_frames,
                save,
                gaussian_var = None)
