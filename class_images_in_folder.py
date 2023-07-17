# ClassInfo = namedtuple("ClassInfo", ["class_name", "class_img_path", "class_image"])    
import cv2

class ClassImagesInFolder:

    @staticmethod
    def generate_from_textfile(text_file_name, images_folder, prefix, suffix):
        with open(text_file_name, "r") as f:
            return {
                i: [
                    class_name.rstrip(), 
                    cv2.imread(str(images_folder / f"{prefix}_{class_name.rstrip()}{suffix}"))
                ]
                for i, class_name in enumerate(f.readlines())
            }

    def __init__(self, class_mapping):
        self._classes = class_mapping

    def get_classes_num(self):
        return len(self._classes)
    
    def get_class_name(self, class_id):
        return self._classes[class_id][0]
    
    def get_images_for_classes(self, classes_list):
        result = {}
        for img_class in classes_list:
            result[img_class] = self._classes[img_class][1]
        return result