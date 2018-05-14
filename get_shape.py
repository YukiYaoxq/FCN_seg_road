import scipy.misc as sc

#path = "/home/yxq/Desktop/um_000000.png"
path ="/home/yxq/Desktop/Kitti-road-semantic-segmentation/data_tiny/data_road/training/image_2/0000160.jpg"
def run():
    image = sc.imread(path)
    shape = image.shape
    print(shape)
run()
