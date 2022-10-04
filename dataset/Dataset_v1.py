from torch.utils.data import Dataset
from PIL import Image
import os
from torchvision import transforms, utils
import random
import csv

class train_val_set(Dataset):
    def __init__(self, subject_img_csv_path, root, val_subject, is_train):
        self._val_subject = val_subject
        self._folder_class_mapping = {
            'c0':0,
            'c1':1,
            'c2':2,
            'c3':3,
            'c4':4,
            'c5':5,
            'c6':6,
            'c7':7,
            'c8':8,
            'c9':9,
        }
        self._subject_img_csv_path = subject_img_csv_path
        self._img_subject_mapping = self.load_image_subject_dict()
        self._root = root
        self._is_train = is_train
        self._train_list, self._val_list = self.create_training_val_list()
        
        
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            # self.normalize
        ])
        self.train_preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224,224)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomAffine(15),
            # self.normalize
        ])

    def load_image_subject_dict(self):
        mapping_dict = {}
        csvFile = open(self._subject_img_csv_path)
        reader = csv.reader(csvFile)
        for item in reader:
            if reader.line_num == 1:
                continue
            mapping_dict[item[2]] = item[0]
        return mapping_dict

    def create_training_val_list(self):
        class_folders = os.listdir(self._root)
        training_list = []
        val_list = []
        for class_folder in class_folders:
            folder_path = os.path.join(self._root,class_folder)
            img_name_list = os.listdir(folder_path)
            for img_name in img_name_list:
                img_path = os.path.join(folder_path,img_name)
                img_class = self._folder_class_mapping[class_folder]
                if self._img_subject_mapping[img_name] not in self._val_subject:
                    training_list.append([img_path,img_class])
                else:
                    val_list.append([img_path,img_class])
        return training_list, val_list
                    
    def load_img(self, path):
        img_pil =  Image.open(path)
        img_pil = img_pil.resize((224,224))
        img_tensor = self.train_preprocess(img_pil)
        return img_tensor

    def load_train_img(self, path):
        img_pil =  Image.open(path)
        img_pil = img_pil.resize((224,224))
        img_tensor = self.preprocess(img_pil)
        return img_tensor
   
    def __getitem__(self, index):
        if self._is_train:
            img_tensor = self.load_train_img(self._train_list[index][0])
            return img_tensor, self._train_list[index][1]
        else:
            img_tensor = self.load_img(self._val_list[index][0])
            return img_tensor, self._val_list[index][1]


    def __len__(self):
        if self._is_train:
            return len(self._train_list)
        else:
            return len(self._val_list)

if __name__ == '__main__':
    val_subject = {'p002','p012','p014'}
    dataset = train_val_set(root = '/media/weiheng/Elements/Data/state-farm-distracted-driver-detection/imgs/train_all', 
    subject_img_csv_path = '/media/weiheng/Elements/Data/state-farm-distracted-driver-detection/driver_imgs_list.csv',
    is_train = False,
    val_subject = val_subject)
    print(len(dataset._train_list))

