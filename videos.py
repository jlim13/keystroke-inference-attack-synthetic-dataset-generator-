import pygame, sys
import random
import os
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
import skimage
import matplotlib.pyplot as plt
from scipy import interpolate
import json
import re
import csv

blur_level = 5
# key press dictionary for users only using their right hands
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
    ' ' : (130, 530),
    'enter' : (230, 530),
    'uppercase' : (0, 500) ,
    'digits' : (25, 530),
    'backspace' : (250, 500)
}


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

def create_blurry_phone(blurred_image, noise_type):

    screen_only_mask = np.load('xr_screen_only_mask.npy')
    to_morph = screen_only_mask == 1
    untouched = screen_only_mask == 0

    original_pix_coords = np.where(screen_only_mask == 0)
    original_pix_values = blurred_image[untouched]


    #add noise only to the screen
    blurred_image = np.asarray(blurred_image, dtype = np.float64)
    #print (np.unique(blurred_image))
    blurred_image *= 1.0/blurred_image.max()
    #print (np.unique(blurred_image))
    blurred_image = skimage.util.random_noise(blurred_image, mode = noise_type) #speckle

    blurred_image = np.asarray(blurred_image * 255.0, dtype = np.uint8)
    blurred_image[zip(original_pix_coords)] = original_pix_values

    blurred_image = Image.fromarray(blurred_image)
    #blurred_image.save('just_noise.png')

    to_darken = ImageEnhance.Brightness(blurred_image)
    dark_level = random.uniform(0.1,1.0)
    blurred_image = to_darken.enhance(dark_level)
    blurred_image.save('xr_blur.png')
    phoneImage = pygame.image.load("xr_blur.png").convert() #change to xr_blur.png

    return phoneImage, blurred_image


class Thumb(object):
    #https://stackoverflow.com/questions/23045685/pygame-move-through-a-list-of-points
    def __init__(self,starting_point, trail):
        self.x= starting_point[0]
        self.y= starting_point[1]
        self.path=trail
        self.atLetter = False
        self.traversal_count = 0
        #x, y is the initial point
        #trail is the remaining letters

    def update(self):
        speed= 8
        self.atLetter = False

        if self.x<(self.path[0])[0]:
            self.x+= speed
        if self.x>(self.path[0])[0]:
            self.x-= speed
        if self.y<(self.path[0])[1]:
            self.y+= speed
        if self.y>(self.path[0])[1]:
            self.y-= speed
        z=(self.x-(self.path[0])[0],self.y-(self.path[0])[1])

        diff = (int(z[0]/-speed), int(z[1]/-speed))

        if diff == (0,0) :

            self.path=self.path[1:]
            ### pause here
            self.atLetter = True
            self.traversal_count +=1

        if self.traversal_count == len(self.path):
            self.atLetter = False


        return self.x, self.y, self.atLetter


def create_words(key_press_dict, word_list):

    word_paths = []

    for word in word_list:
        path = [list(key_press_dict[x]) for x in word]
        word_paths.append([word, path])

    return word_paths


def create_thumb(blurred_image, add_noise, left_handed, add_blit, noise_type ):


    #let's create a surface to hold our ellipse:
    surface = pygame.Surface((230, 90),pygame.SRCALPHA)

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

    thumb_color = random.choice(colors)
    size = (0, 0, 230, 90)

    #if we want to add some noise to the thumb
    if add_blit:
        pygame.draw.ellipse(surface, (25,25,25), (0,0,285,130))

    ellipse = pygame.draw.ellipse(surface, thumb_color, size)

    #new surface variable for clarity (could use our existing though)

    rand_noise = random.uniform(-5, 5)
    noisy_thumb = -75.  + rand_noise

    surface2 = pygame.transform.rotate(surface, noisy_thumb)

    ###### Create a left thumb
    if left_handed:
        surface2 = pygame.transform.flip(surface2, True, False)

    ###### Blur the thumb
    surface2 = blurSurf(surface2, 7.0)
    key, noise_coords = gen_key(key_press_dict) #dont need this for words

    return surface2

def create_videos(screen, thumb_surface, word_path, phoneImage,save_dir, save = False):

    running = True

    thumb_path = Thumb(word_path[0], word_path[1:])
    frame_count = 0
    clock = pygame.time.Clock()

    while running:
        clock.tick(60)
        screen.fill((225 ,225, 215))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()


        screen.blit(phoneImage, (0,0))

        try:
            new_x, new_y, atLetter = thumb_path.update()

            screen.blit(thumb_surface, [new_x, new_y]) #location of ellipse

        except:
            #word is done

            running = False

        # regular thumb gen

        pygame.display.update()
        frame_count += 1
        frame_to_save_dest = os.path.join(save_dir, str(frame_count)+ '.png')

        if save:
            if atLetter:
                #if at letter, save some extra frames
                extra_frames = random.randint(1,5)
                extra_frames = 0
                pygame.display.update()
                for i in range(extra_frames):
                    frame_count += 1
                    frame_to_save_dest = os.path.join(save_dir, str(frame_count)+ '.png')
                    pygame.image.save(screen, frame_to_save_dest)

                pygame.display.update()
            else:
                pygame.image.save(screen, frame_to_save_dest)

    running = True
    print (frame_count)
    frame_count = 0



def parse_word_list(csv_file):
    phrase_list = []
    with open(csv_file, newline='') as csvfile:

        spamreader = csv.reader(csvfile, delimiter=',' )
        count = 0
        next(spamreader)
        for idx, row in enumerate(spamreader):
            date, phrase = row
            count += 1
            phrase = ''.join(phrase)
            phrase =  re.sub('[^a-zA-Z]+', ' ', phrase).lower()
            phrase_list.append(phrase)

    return phrase_list

def main_(  phrase_list,
            add_noise,
            add_blit,
            noise_type,
            save,
            output_dir,
            xr_im,
            left_handed = False):


    instances = 1
    phrase_list = parse_word_list(phrase_list)

    phrase_list = [x.lower() for x in phrase_list]
    screen = pygame.display.set_mode((312, 632))
    running = True
    im = Image.open(xr_im)

    blurred_image = im.filter(ImageFilter.GaussianBlur(radius = blur_level))
    blurred_image_np = np.array(blurred_image)

    paths = create_words(key_press_dict, phrase_list)

    total_files = len(phrase_list) * instances
    completed = 0

    for word, (_, path) in zip(phrase_list, paths):

        blurred_image = Image.fromarray(blurred_image_np)
        to_darken = ImageEnhance.Brightness(blurred_image)
        dark_level = random.uniform(0.1,1.0)
        blurred_image = to_darken.enhance(dark_level)
        blurred_image.save('xr_blur.png')
        phoneImage = pygame.image.load("xr_blur.png").convert() #change to xr_blur.png

        print ("Completed {} out of {} ".format(completed, total_files))
        if add_noise:
           phoneImage, _ = create_blurry_phone(blurred_image,
                                                noise_type)

        save_dir = os.path.join(output_dir, word)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)


            thumb_surface = create_thumb(blurred_image,
                                        add_noise,
                                        left_handed,
                                        add_blit,
                                        noise_type)
            create_videos(screen, thumb_surface, path, phoneImage, save_dir, save)
        completed += 1


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()


    parser.add_argument("--phrase_list_file", type = str, default = '/data/jlim/abcnews-date-text.csv' )
    parser.add_argument("--add_noise", type = str2bool, default = False )
    parser.add_argument("--add_blit", type = str2bool, default = False)
    parser.add_argument("--left_handed", type = str2bool, default = False)
    parser.add_argument("--noise_type", type = str, default = 'gaussian')
    parser.add_argument("--output_dir", type = str, default =  'video_output/')
    parser.add_argument("--xr_im", type = str, default = 'xr_.png')
    parser.add_argument("--save", type = str2bool, default = True )


    args = parser.parse_args()


    params = main_(args.phrase_list_file,
                    args.add_noise,
                    args.add_blit,
                    args.noise_type,
                    args.save,
                    args.output_dir,
                    args.xr_im,
                    args.left_handed)
