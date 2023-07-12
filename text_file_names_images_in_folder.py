class TextFileNamesImagesInFolder:
    def __init(self, text_file_name, images_folder):
        self._text_file_name = text_file_name
        self._images_folder = images_folder
        

    def get_classes_num(self):
        return len(self._classes)
    
    def get_images_for_classes(self, classes_list):
        result = []
        for img_class in classes_list:
            result.append(self._classes[img_class].image)
        return result            