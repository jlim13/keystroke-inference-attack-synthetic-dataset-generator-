import numpy as np
import matplotlib.pyplot as plt

class Key:
    def __init__(self, char, image_crop, bounding_box, center):
        self.char = char
        self.image_crop = image_crop
        self.bounding_box = bounding_box
        self.center = center

class Keyboard:

    def __init__(self, type, im_file_name, key_to_info_dict, dsitribution = None):

        self.type = type
        self.im_file_name = im_file_name
        self.key_to_info_dict = key_to_info_dict
        self.key_list = {}


    def _center_location(self, bounding_box):
        x_min, x_max, y_min, y_max = bounding_box
        center_x = x_min + ((x_max - x_min)/2)
        center_y = y_min + ((y_max - y_min)/2)
        return center_x, center_y

    def reconstruct(self, type):
        assert(self.type == 'reconstruct')

    def view_keyboard(self):
        if isinstance(self.im_file_name, str):
            plt.imshow(plt.imread(self.im_file_name))
            plt.show()
        else:
            plt.imshow(self.im_file_name)
            plt.show()



    def get_key(self, char):
        '''
        input = key (i.e. 'a')
        output = the image crop, bounding box, key center
        '''
        key_information = self.key_to_info_dict[char]

        image_crop = key_information[0]
        bounding_box = key_information[1]
        reference_letter = key_information[2]

        center_x, center_y = self._center_location(bounding_box)
        key = Key(char, image_crop, bounding_box, (center_x, center_y))
        self.key_list[char] = key
        return key
