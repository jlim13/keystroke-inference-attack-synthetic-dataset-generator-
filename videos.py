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
            save):


    instances = 1
    phrase_list = parse_word_list(phrase_list)

    phrase_list = ["the rainfall during the spring is brutal",
                "the notebook is over there",
                "its much more difficult to play tennis with a bowling ball",
                "the laptop is out of battery",
                "orange is my favorite juice",
                "the high school yearbook is down by the shelf",
                "thanks so much for the help",
                "she did not cheat on the test for it was not the right thing to do",
                "the clock within this blog and the clock on my laptop are different",
                "what channel is that game on later",
                "i am going to the store",
                "i work for a large company",
                "why wont you listen",
                "at that moment she realized she had a sixth sense",
                "he swore he just saw his sushi move",
                "which bank do you use",
                "looks like its going to rain",
                "the quick brown fox jumped over the lazy dog",
                "i have plans with my friend but we can go tomorrow",
                "too many prisons have become early coffins",
                "the barcelona game is on later tonight",
                "clean the floor please",
                "there is a good amount of debris outside",
                "i will schedule an appointment with the dentist",
                "my internet is so slow today",
                "do you like sports",
                "can you go pick my mail up at the mailbox",
                "how was your exam",
                "beautiful day for golf dont you think",
                "k getting on",
                "at least it looks nice",
                "i am deciding between rugby cricket and golf",
                "i think i need a belt because i lost weight",
                "i went to go milk the cow",
                "what text editor do you use",
                "even with the snow falling outside",
                "you will injure your head doing that",
                "singapore is so hot and humid",
                "he strives to keep the best lawn in the neighborhood",
                "what a good way to spend your last day",
                "i have been taking lessons for different sports",
                "i need to customize my shoes",
                "have to make a call",
                "i am not sure where you are going",
                "what is your schedule like for the next week",
                "i am ready for bed",
                "asheville is nice this time of year",
                "november third is election day",
                "he is no james bond his name is roger moore",
                "can i please come to the concert with you",
                "my schoolwork is going well",
                "yeah i could probably play tonight",
                "i just got new golf clubs today",
                "i cant wait until football is back",
                "that is a good summary",
                "you need to stretch your back out",
                "what a rare find",
                "rugby seems like a fun sport",
                "i play tennis tonight",
                "i need to eat dinner",
                "i have an appointment with the vet now",
                "when can i come over",
                "be careful with that butter knife",
                "he decided water skiing on a frozen lake wasnt a good idea",
                "are you sick",
                "my internship is almost done",
                "what did you think of the game",
                "you bite up because of your lower jaw",
                "queensland is a place on the map",
                "what time is the movie",
                "what concert are you going to tonight",
                "i have no internet or power because of the storm",
                "look at my hair",
                "my golf lesson is going pretty poorly",
                "the fridge is full",
                "its not possible to convince a monkey to give you a bannana",
                "the ruler is over there",
                "what is your favorite travel destination",
                "i just got a new phone today",
                "i will be heading out to the golf course tomorrow",
                "i have so many books to read",
                "how was your senior year",
                "i am so excited to show you my new phone",
                "my girlfriend liked the gift",
                "what courses will you be taking",
                "thanks so much for all the help on the test",
                "i need to go to the store",
                "let me finish this work",
                "the sun is a shade of yellow",
                "the flight to singapore is so long",
                "lets start a bookclub",
                "want to get some coffee after this",
                "what is your favorite juice flavor",
                "I have a cleaner coming in tomorrow",
                "can you facetime me later",
                "making dinner now",
                "the professor is going to cover a new topic",
                "that was a good football game",
                "i am taking a class",
                "it was obvious she was hot sweaty and tired",
                "no watching a show",
                "water is good for you",
                "thank you for your time earlier",
                "school is hard",
                "how is your credit score these days",
                "pat ordered a ghost pepper pie",
                "the flower looks nice",
                "my wallet is out of money",
                "nice photo",
                "please clean up after yourself",
                "hello",
                "okay thanks for preparing foods",
                "how can you eat that",
                "what was the intention for that shot",
                "ok can you give me five",
                "now i have a tv",
                "the photo album is complete",
                "how did your interview go last week",
                "hello how are you",
                "my dog needs a haircut",
                "what time are you going to be playing",
                "our team made a great comeback earlier",
                "big crowds stress me out",
                "it was sunny earlier today",
                "i started to take a course",
                "when can i see you",
                "your package came",
                "seek success but always be prepared for random cats",
                "was there a warranty",
                "come again",
                "i wish i could have answered the questions better",
                "she had not had her cup of coffee",
                "i am playing a video game",
                "jack was reading me the book",
                "i did not understand the teacher",
                "just finishing something up",
                "the trash can is over there",
                "i am stting on this desk",
                "we are going to the library later",
                "how is the weather",
                "i am at the location",
                "i am stuck on my homework assignment",
                "like to drink water",
                "he was telling to go there",
                "my dog likes to eat my eggs in the morning",
                "i appreciate the gift",
                "the course is a bit boring",
                "what did you eat for dinner",
                "where is the mailbox",
                "my dog needs to get his haircut",
                "i have to go to the dentist",
                "where is my dog getting his bone",
                "this game",
                "she wondered what his eyes were saying",
                "the bees decided to have a mutiny against their queen",
                "the store was closed",
                "how was your day at work",
                "take your pills later today",
                "i like to play catan",
                "what does that word mean",
                "can you not",
                "she let the balloon float up into the air",
                "thank you for the idea",
                "how is your course coming along",
                "the course was pretty good",
                "australia is a hot place",
                "where can i get help for that",
                "thank you for your tim",
                "please check your messages later",
                "do you watch cricket with your friends",
                "what time is your show",
                "your order is ready",
                "python is a good language",
                "i have been to singapore",
                "i wish i could go to the pool later today",
                "i currently have four windows open",
                "i need to start doing my homework on time",
                "okay sounds good",
                "my dogs name is rockie",
                "turkey is for dinner",
                "the teacher called my parents last night",
                "i am sorry to hear that",
                "my laptop needs a charge",
                "i think we are lost go ask for directions",
                "the controller is so far away",
                "the shirt looks bad on you",
                "that sounds like a great idea",
                "i like to eat food",
                "like to play basketball",
                "i cant believe our team lost like that",
                "the paper was boring",
                "why would you do that",
                "what is going on with your mom",
                "my pen ran out of ink",
                "i studied abroad in singapore for a semester",]
    phrase_list = [x.lower() for x in phrase_list]
    screen = pygame.display.set_mode((312, 632))
    running = True
    im = Image.open("xr_.png")

    blurred_image = im.filter(ImageFilter.GaussianBlur(radius = blur_level))
    blurred_image_np = np.array(blurred_image)

    # if not add_noise:

        # blurred_image = Image.fromarray(blurred_image)
        # to_darken = ImageEnhance.Brightness(blurred_image)
        # blurred_image = to_darken.enhance(0.17)
        # blurred_image.save('xr_blur.png')
        # phoneImage = pygame.image.load("xr_blur.png").convert() #change to xr_blur.png

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

        save_dir = os.path.join('vid2vid_input_syn/', word)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

            left_handed = False
            thumb_surface = create_thumb(blurred_image,
                                        add_noise,
                                        left_handed,
                                        add_blit,
                                        noise_type)
            create_videos(screen, thumb_surface, path, phoneImage, save_dir, save)
        completed += 1




if __name__ == "__main__":

    phrase_list_file = '/data/jlim/abcnews-date-text.csv'
    add_noise = False
    add_blit = False
    noise_type = 'shadow'
    noise_type = 'gaussian'

    save = True

    params = main_(phrase_list_file, add_noise, add_blit, noise_type, save)
