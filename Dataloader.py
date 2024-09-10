import os
from torch.utils.data import Dataset
from torchvision.io import read_image
import matplotlib.pyplot as plt
import matplotlib.patches as patches  # Ensure this is imported

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.img_labels = [file_name for file_name in os.listdir(img_dir) if file_name.endswith(('.txt'))]
        self.img_dir = img_dir
        # self.transform = transform
        # self.target_transform = target_transform
    def __len__(self): # load len(txt files)
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        txt_file_name = self.img_labels[idx]
        jpg_file_name = os.path.splitext(txt_file_name)[0] + '.jpg'
        img_path = os.path.join(self.img_dir, jpg_file_name)
        image = read_image(img_path)
        # read txt file
        with open(os.path.join(self.img_dir, txt_file_name), 'r') as file:
            data = list(filter(None, file.read().split()))
            print('data',data)
            # data = list(filter(None, file.read().strip().split()))
            # print('data',data)
        # make dictionary
        label_dict = {}
        # class
        label_dict['class'] = data[0]
        # face bbox x, y, w, h
        label_dict['x'] = float(data[1])
        label_dict['y'] = float(data[2])
        label_dict['w'] = float(data[3])
        label_dict['h'] = float(data[4])
        # landmark
        points = []
        for i in range(5, len(data), 3):  # 2.0000을 건너뛰기 위해 3씩 증가
            x_coord = float(data[i])
            y_coord = float(data[i + 1])
            points.append((x_coord, y_coord))
        label_dict['points'] = points
        # if self.transform: ->
        #     image = self.transform(image)
        # if self.target_transform:
        #     label = self.target_transform(label) -> 0-255 -> float으로 바꾸고 -1-1사이로, 전처리 
        return image, label_dict
    
    def __visualize__(self, idx):
        # Get image and label
        image, label_dict = self.__getitem__(idx)
        # Convert image to numpy array (if it's not already)
        image = image.permute(1, 2, 0).numpy()  # Change from (C, H, W) to (H, W, C)
        image = image.astype('uint8')  # Ensure the image is in uint8 format
        img_height, img_width = image.shape[:2]
        center_x = label_dict['x'] * img_width
        center_y = label_dict['y'] * img_height
        width = label_dict['w'] * img_width
        height = label_dict['h'] * img_height
        # Calculate the top-left corner of the bounding box
        x = center_x - (width / 2)
        y = center_y - (height / 2)
        # Plot image
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(image)
        # Draw bounding box
        bbox = patches.Rectangle(
            (x, y),
            width,
            height,
            linewidth=2,
            edgecolor='r',
            facecolor='none'
        )
        ax.add_patch(bbox)
        # Draw landmarks
        for (x_coord, y_coord) in label_dict['points']:
            x_coord *= img_width
            y_coord *= img_height
            ax.plot(x_coord, y_coord, 'bo')  # Blue dots for landmarks
        plt.show()
img_dir = '/Users/hong-eun-yeong/Desktop/Seminar/_etc/'
# Example usage
dataset = CustomImageDataset(img_dir=img_dir)
print((dataset[0]))
dataset.__len__()
# Visualize the first image in the dataset
# Visualize specific indices
indices_to_visualize = [0]  # Replace with the indices you want to visualize
for idx in indices_to_visualize:
    dataset.__visualize__(idx)











