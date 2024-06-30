from src.dataset.kitti_loader_3D.Dataloader.bin2depth import get_calibration_files
from src.dataset.kitti_loader_3D.dataset_3D import DataGenerator

train_gen = DataGenerator(KittiDir=r"C:\Users\jerem\OneDrive\Me\StudiumMaster\00_Semesterarbeit\Project\MMSiamese\data\kitti", phase='train')
train_loader = train_gen.create_data(32, shuffle=True)

for batch in train_loader:
    velo = batch['velo']
    path = batch['cam_path'][0]
    a = get_calibration_files(calib_dir=path)
    # print(type(path))
# from src.dataset.kitti_loader_3D.Dataloader import Kittiloader
#
# loader = Kittiloader(kittiDir=r"C:\Users\jerem\OneDrive\Me\StudiumMaster\00_Semesterarbeit\Project\MMSiamese\data\kitti", mode="train")
#
# data = loader.load_item(idx=0)
# data2 = loader.load_item(idx=10)
#
# # print(data)
# # print(data2)
# print((data["velo"].shape))
# print((data2["velo"].shape))