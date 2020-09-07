from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import random
import numpy as np
from scipy import spatial, stats
import xml.etree.ElementTree as ET
import re, string
import operator
import scipy
import random
from keyboard import Keyboard
import os
import json
import ast

random.seed(4)

sms_file = '../data/nus-sms-corpus/smsCorpus_en_2015.03.09_all.xml'

def randomize_keyboard(original_im, key_dict,random_im_name = 'random_keyboard_xr.png'):

    im = plt.imread(original_im)
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

    # plt.imshow(im)
    # plt.scatter(x = xs, y =ys)
    # plt.show()

    keys = list(bboxes.keys())
    new_keys = keys[:]
    #print (keys)

    random.shuffle(new_keys)
    #print (new_keys)

    new_bboxes = {}

    for i in range(len(keys)):
        key = keys[i]
        new_key = new_keys[i]
        #new key is a key that
        new_bboxes[new_key] = bboxes[key]

    #print ("===============")

    prev_crops = []
    c = 0

    new_dict = key_dict.copy()

    for k,v in new_bboxes.items():
        new_key = k
        old_key = v[2]
        new_dict[old_key] = key_dict[new_key]


    key_2_bb_dict = {}

    for k, v in new_bboxes.items():
        crop = v[0]


        for c in prev_crops:
            if ( (c[0]==crop).all()):
                print (k)
                print (c[1])

                a = new_bboxes[k][0]
                b = new_bboxes[c[1]][0]


        prev_crops.append([crop, k])

        bbox = bboxes[k][1] # bounding box for old one that needs to get replaced by new letter
        x_min, x_max, y_min, y_ymax = bbox[0], bbox[1], bbox[2], bbox[3]
        new_im[y_min:y_min+32, x_min:x_min+24, :] = crop

        #this dict maps a key to its new bounding box
        key_2_bb_dict[v[2]] = [bbox, crop]


    # plt.imshow(new_im)
    # plt.show()

    new_im *= 255
    new_im = np.array(new_im, dtype = np.uint8)
    im = Image.fromarray(new_im)
    im.save(random_im_name)



    return random_im_name, new_dict, key_2_bb_dict

def center_loc(x_min, x_max, y_min, y_max):
    center_x = x_min + ((x_max - x_min)/2)
    center_y = y_min + ((y_max - y_min)/2)
    return center_x, center_y

def process_text_file(txt_file = 'common_words.txt'):
    words = []

    with open(txt_file) as f:
        for line in f:
            line = line.strip('\n')
            line = [char for char in line]
            words.append(line)

    return words

def parseXML(xmlfile = sms_file):
    texts = []
    # create element tree object
    tree = ET.parse(xmlfile)
    # get root element
    root = tree.getroot()
    # print ([elem.tag for elem in root.iter()])
    for txt in root.iter('text'):
        sms = txt.text
        words = re.sub('[^a-zA-Z]+', ' ', sms).split(' ')
        words = [x.lower() for x in words]
        for word in words:
            if word:
                texts.append(word.lower())

    #random.shuffle(texts)
    return texts

def center_noise(key, bboxes):

    bbox = bboxes[key][0]
    x_min, x_max, y_min, y_max = bbox
    bbox_center = center_loc(x_min, x_max, y_min, y_max)

    rn_x = random.uniform(-3, 3)
    rn_y = random.uniform(-4.5, 4.5)

    adjust_center = [round(bbox_center[0] + rn_x), round(bbox_center[1] + rn_y)]

    return adjust_center

def nearest_key(noisy, key_centers):
    #this maps a noisy key to key center

    #key_centers = list(key_centers.keys())
    tree = spatial.KDTree(key_centers)
    distance, idx = tree.query(noisy)
    nearest_pt = key_centers[idx]


    return nearest_pt

def match_distributions(words, key_dict, random_im, new_dict, bboxes, word_limit):

    #words = process_text_file()
    original_distribution = {
                    'e' : 12.012,
                    't' : 9.10,
                    'a' : 8.12,
                    'o' : 7.68,
                    'i' : 7.31,
                    'n' : 6.95,
                    's' : 6.28,
                    'r' : 6.02,
                    'h' : 5.92,
                    'd' : 4.32,
                    'l' : 3.98,
                    'u' : 2.88,
                    'c' : 2.71,
                    'm' : 2.61,
                    'f' : 2.30,
                    'y' : 2.11,
                    'w' : 2.09,
                    'g' : 2.03,
                    'p' : 1.82,
                    'b' : 1.49,
                    'v' : 1.11,
                    'k' : .69,
                    'x' : .17,
                    'q' : .11,
                    'j' : .10,
                    'z' : .07}

    freq_map = {}
    #maps pixel location to frequencies

    for idx, word in enumerate(words):
        if idx == word_limit:
            break

        center_map = [center_noise(x, bboxes) for x in word]

        for pix in center_map:
            pix = tuple(pix)
            if pix not in freq_map:
                freq_map[pix] = 1
            else:
                freq_map[pix] += 1

    '''
    First, bin the keys ((x,y) coordinates) so that we get 26 keys
    Finally, output the frequency
    '''
    bbox_to_center = {}

    for key, bbox in bboxes.items():
        bbox = bboxes[key]
        x_min, x_max, y_min, y_max = bbox[0]
        bbox_center = center_loc(x_min, x_max, y_min, y_max)
        bbox_to_center[bbox_center] = (0, key)


    freq_cts = {}
    center_2_bbox = {}
    key_centers = list(bbox_to_center.keys())

    total_c = 0
    for loc, ct in freq_map.items():
        # bin the noisy key into one of 26 bins
        nearest_pt = nearest_key(loc, key_centers)
        if nearest_pt not in freq_cts:
            freq_cts[nearest_pt] = ct
            #center_2_bbox[nearest_pt] =
        else:
            freq_cts[nearest_pt] += ct
        total_c += ct

    '''
    Map the key center to letter
    '''

    freq_dict = {}

    total_words = 0.
    for k,v in freq_cts.items():
        letter = bbox_to_center[k][1]
        total_words += v
        freq_dict[letter] = v

    print (freq_dict)
    exit()

    recon_keyboard, observed_dist = reconstruct_keyboard(freq_cts,
                        original_distribution,
                        bbox_to_center,
                        bboxes)


    for k,v in freq_cts.items():
        letter = bbox_to_center[k][1]
        freq_dict[letter] = round((freq_dict[letter]/total_words) * 100., 2)

    sorted_original = sorted(original_distribution.items(),
                        key = operator.itemgetter(1), reverse = True)
    sorted_observed = sorted(freq_dict.items(),
                        key = operator.itemgetter(1), reverse = True)

    original_dist = np.zeros(26)
    observed_dist = np.zeros(26)

    for idx, (orig, obs) in enumerate(zip(sorted_original, sorted_observed)):
        original_dist[idx] = orig[1]
        observed_dist[idx] = obs[1]

    return recon_keyboard, sorted_observed, key_centers

def reconstruct_keyboard(freq_cts, original_distribution, bbox_to_center, bboxes):
    #tries to reconstruct observed keyboard


    im = plt.imread('random_keyboard_xr.png')

    observed_dist = sorted(freq_cts.items(), key=operator.itemgetter(1), reverse= True)
    observed_dict = {}

    recon_keyboard_dict = {}

    for idx, hyp in enumerate(observed_dist):

        center_h, count_h = hyp
        letter_h = list(original_distribution.keys())[idx]
        letter_real = bbox_to_center[center_h][1]

        info_hyp = bboxes[letter_h]
        bb_hyp = info_hyp[0]
        crop_hyp = info_hyp[1]

        bb_real = bboxes[letter_real][0]
        x_min, x_max, y_min, y_max = bb_real  # NEED TO GET REAL BB
        im[y_min:y_min+32, x_min:x_min+24, :] = crop_hyp
        recon_keyboard_dict[letter_h] = [crop_hyp, bb_real, letter_real]

        observed_dict[center_h] = {'hyp' :letter_h, 'gt': letter_real}


    recon_keyboard = Keyboard(type='recon',
                                im_file_name = im,
                                key_to_info_dict = recon_keyboard_dict)

    return recon_keyboard, observed_dict

def hamming_distance(s1, s2):
    """Return the Hamming distance between equal-length sequences"""
    if len(s1) != len(s2):
        raise ValueError("Undefined for sequences of unequal length")
    cost = 0
    incorrect_letters = []
    for el1, el2 in zip(s1,s2):
        if el1 != el2:
            cost +=1
            incorrect_letters.append(el2)

    return cost, incorrect_letters

def score_words(recon_keyboard, common_words, ground_truth_keyboard):


    gt_board = ground_truth_keyboard.key_to_info_dict
    pred_board = recon_keyboard.key_to_info_dict

    best_score = -1
    '''
    #iterate through all words in common_words
    for word in common_words:
        "type" that word in my keyboard


    '''

    score = 0
    common_words = common_words[:100]
    for word in common_words:
        reconstructed_word = ''
        for letter in word:

            gt_key_info = gt_board[letter]
            gt_key = gt_key_info[0]
            gt_key_bbox = gt_key_info[1]
            pred_key_info = pred_board[letter]
            pred_key = pred_key_info[1]
            pred_key_bbox = pred_key_info[0]
            real_letter = pred_key_info[2]
            reconstructed_word += real_letter
            #I think I pressed on "letter" when I actually pressed "real_letter"

        word = ''.join(word)
        recon_word = ''.join(reconstructed_word)

        if recon_word in common_words:
            score +=1
    print (score)
    '''
    if score is less than
    '''

def parseWiki():
    wiki_dir = '../data/wiki_text/wikiextractor/wiki_text/AA/'
    all_lines = []

    for wiki_article in os.listdir(wiki_dir):
        wiki_article_path = os.path.join(wiki_dir, wiki_article)
        with open (wiki_article_path, encoding = 'utf-8') as wiki_f:
            for idx, line in enumerate(wiki_f):
                if idx == 0: continue
                all_lines.append(line)

    texts = []

    for idx, line in enumerate(all_lines):

        words = re.sub('[^a-zA-Z]+', ' ', line).split(' ')
        words = [x.lower() for x in words]
        for word in words:
            if word:
                texts.append(word.lower())

        # if idx == 100:
        #     break
    return texts


key_dict  = {
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

random_im, new_dict, bboxes = randomize_keyboard('xr_.png', key_dict )

ground_truth_keyboard = Keyboard(type = 'random',
                                im_file_name = random_im,
                                key_to_info_dict = bboxes)

word_limits = [  1000, 5000, 10000, 15000, 20000, 10000000]

common_words = process_text_file()
words = parseWiki()


for word_limit in word_limits:
    word_limt = 10000000000000
    recon_keyboard, observed_dict, key_centers = match_distributions(words,
                                        key_dict,
                                        random_im,
                                        new_dict,
                                        bboxes,
                                        word_limit)

    #ground_truth_keyboard.view_keyboard()
    #recon_keyboard.view_keyboard()
    print (observed_dict)
    score_words(recon_keyboard, common_words, ground_truth_keyboard)

    exit()

    #
    # #with new random keyboard, predict words
    #
    # total_cost = 0
    #
    # all_incorrect_letters = []
    #
    # # keep sampling n words until I get the new layout
    #
    # dist = 10000
    #
    # for i in range(100):
    #     sampled_words = random.shuffle(common_words)
    #     for idx, word in enumerate(sampled_words):
    #         if idx == 1999: break
    #         center_map = [center_noise(x, bboxes) for x in word]
    #
    #         observed_word = ''
    #
    #         for pix in center_map:
    #             pix = tuple(pix)
    #             nearest_pt = nearest_key(pix, key_centers)
    #             hyp_l = observed_dict[nearest_pt]['hyp']
    #             observed_word += hyp_l
    #
    #         actual_word = ''.join(word)
    #         samp_dist, incorrect_letters = hamming_distance(actual_word, observed_word)
    #
    #         if samp_dist < min_dist:
    #             dist = samp_dist
    #
    #
    #         # total_cost += dist
    #         #all_incorrect_letters.append(incorrect_letters)
    #
    # #all_incorrect_letters = set([item for sublist in all_incorrect_letters for item in sublist])
    # # print ("Hamming distance {0}".format(total_cost/idx))
    # #print ("The incorrect letters are: {0}".format(all_incorrect_letters))
    # # print ("When we observe {0} words, we create a keyboard with {1} errors".format(word_limit, len(all_incorrect_letters)))
