import os  
import random  
import shutil  

def create_dataset(image_folder,dsm_folder,dataset_folder):
    # 原始文件夹路径
    ## image_folder,dsm_folder
    
    # 训练集和测试集文件夹路径
    train_image_folder = dataset_folder + '/train/image'
    train_dsm_folder = dataset_folder + '/train/dsm'
    test_image_folder = dataset_folder + '/test/image'
    test_dsm_folder = dataset_folder + '/test/dsm'
    #确保文件夹存在
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
        os.makedirs(dataset_folder+'/train')
        os.makedirs(train_image_folder)
        os.makedirs(train_dsm_folder)
        os.makedirs(dataset_folder+'/test')
        os.makedirs(test_image_folder)
        os.makedirs(test_dsm_folder)
        
    # 获取所有文件的列表  
    image_files = [f for f in os.listdir(image_folder)]
    random.shuffle(image_files)
    
    # 划分训练集和测试集 
    split_index = int(len(image_files) * 0.8)  
    train_files = image_files[:split_index]  
    test_files = image_files[split_index:] 
    
    # 将文件移动到对应的文件夹中 
    for file in train_files:
        shutil.move(os.path.join(image_folder, file), os.path.join(train_image_folder, file))
        dsm_file = file[:-3] + "tif"
        shutil.move(os.path.join(dsm_folder, dsm_file), os.path.join(train_dsm_folder, dsm_file))
    for file in test_files:
        shutil.move(os.path.join(image_folder, file), os.path.join(test_image_folder, file))
        dsm_file = file[:-3] + "tif"
        shutil.move(os.path.join(dsm_folder, dsm_file), os.path.join(test_dsm_folder, dsm_file))
        
    print(f"Moved {len(train_files)} files to train")  
    print(f"Moved {len(test_files)} files to test")
    
if __name__ == "__main__":
    create_dataset(
        image_folder="./IEEE_data/rgbd",
        dsm_folder="./IEEE_data/dsm",
        dataset_folder="./IEEE_data/dataset",
    )