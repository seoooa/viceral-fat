import autorootcwd
from monai.transforms import (
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    RandFlipd,
    CropForegroundd,
    Compose,
    Spacingd,
    AsDiscreted,
    GaussianSmoothd,
    Lambda,
)
from monai.data import CacheDataset, DataLoader, Dataset
import os
import SimpleITK as sitk
import torch
import lightning.pytorch as pl
from pathlib import Path
import yaml

def create_distance_map(binary_mask):
    """
    create distance map from binary mask
    
    Args:
        binary_mask (torch.Tensor): [C, H, W, D] binary mask
        
    Returns:
        torch.Tensor: [C, H, W, D] distance map
    """
    distance_maps = []
    
    for c in range(binary_mask.shape[0]):  # each channel
        channel_mask = binary_mask[c].numpy()  # [H, W, D]
        
        # convert to SimpleITK image
        sitk_mask = sitk.GetImageFromArray(channel_mask)
        sitk_mask = sitk.Cast(sitk_mask, sitk.sitkUInt8)
        
        # create distance map
        distance_map = sitk.SignedMaurerDistanceMap(
            sitk_mask,
            insideIsPositive=False,  # heart outside is positive
            squaredDistance=False,
            useImageSpacing=True     # physical distance (mm)
        )
        
        # convert to tensor
        distance_map_array = sitk.GetArrayFromImage(distance_map)
        distance_maps.append(torch.from_numpy(distance_map_array))
    
    return torch.stack(distance_maps)

def ConvertDistanceMap(data):
    """
    convert segmentation to distance map
    """
    seg = data["seg"]  # [C, H, W, D] one-hot encoded segmentation
    
    distance_map = create_distance_map(seg)
    
    data["seg"] = distance_map
    return data

class ViceralFatDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/KU-PET-CT",
        batch_size: int = 4,
        patch_size: tuple = (96, 96, 96),
        num_workers: int = 4, 
        cache_rate: float = 0.1,
        use_distance_map: bool = False,
        fold_number: int = 1
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.cache_rate = cache_rate
        self.use_distance_map = use_distance_map
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self.fold_number = fold_number

    def load_data_splits(self, yaml_path, fold_number):
        # Read the YAML file
        with open(yaml_path, "r") as file:
            data = yaml.safe_load(file)
        
        # Extract train, val, test splits from the specified fold
        fold_key = f"fold_{fold_number}"
        fold = data["cross_validation_splits"][fold_number-1][fold_key]
        train_split = fold["train"]
        val_split = fold["val"]
        test_split = fold["test"]

        # Add folder path to each entry in the splits and create dictionaries
        base_dir = os.path.dirname(yaml_path)
        train_split = [
            {
                "image": os.path.join(base_dir, entry, "CT.nii.gz"),
                "label": os.path.join(base_dir, entry, "vf.nii.gz"),
                "seg": os.path.join(base_dir, entry, "combined_seg.nii.gz"),
            }
            for entry in train_split
        ]
        val_split = [
            {
                "image": os.path.join(base_dir, entry, "CT.nii.gz"),
                "label": os.path.join(base_dir, entry, "vf.nii.gz"),
                "seg": os.path.join(base_dir, entry, "combined_seg.nii.gz"),
            }
            for entry in val_split
        ]
        test_split = [
            {
                "image": os.path.join(base_dir, entry, "CT.nii.gz"),
                "label": os.path.join(base_dir, entry, "vf.nii.gz"),
                "seg": os.path.join(base_dir, entry, "combined_seg.nii.gz"),
            }
            for entry in test_split
        ]
        print(f"Loaded data splits from {yaml_path} for fold {fold_number}")

        return train_split, val_split, test_split

    def prepare_data(self):
        transforms = [
            LoadImaged(keys=["image", "label", "seg"]),
            EnsureChannelFirstd(keys=["image", "label", "seg"]),
            Orientationd(keys=["image", "label", "seg"], axcodes="RAS"),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-200,
                a_max=100,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            AsDiscreted(
                keys=["seg"],
                to_onehot=8,
            ),
            CropForegroundd(keys=["image", "label", "seg"], source_key="image"),
        ]

        # distance map 
        if self.use_distance_map:
            transforms.append(Lambda(ConvertDistanceMap))

        transforms.extend([
            RandCropByPosNegLabeld(
                keys=["image", "label", "seg"],
                label_key="label",
                spatial_size=(96, 96, 96),
                pos=1,
                neg=1,
                num_samples=4,
                image_key="image",
                image_threshold=0,
            ),
            RandFlipd(
                keys=["image", "label", "seg"],
                spatial_axis=[0],
                prob=0.10,
            ),
            RandFlipd(
                keys=["image", "label", "seg"],
                spatial_axis=[1],
                prob=0.10,
            ),
        ])

        if not self.use_distance_map:
            transforms.append(GaussianSmoothd(keys=["seg"], sigma=1.0))

        transforms.append(RandShiftIntensityd(keys="image", offsets=0.05, prob=0.5))

        self.train_transforms = Compose(transforms)

        val_transforms = [
            LoadImaged(keys=["image", "label", "seg"]),
            EnsureChannelFirstd(keys=["image", "label", "seg"]),
            Orientationd(keys=["image", "label", "seg"], axcodes="RAS"),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-200,
                a_max=100,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            AsDiscreted(
                keys=["seg"],
                to_onehot=8,
            ),
            CropForegroundd(keys=["image", "label", "seg"], source_key="image"),
        ]

        # distance map
        if self.use_distance_map:
            val_transforms.append(Lambda(ConvertDistanceMap))

        if not self.use_distance_map:
            val_transforms.append(GaussianSmoothd(keys=["seg"], sigma=1.0))

        self.val_transforms = Compose(val_transforms)

    def setup(self, stage=None):
        train_files, val_files, test_files = self.load_data_splits(
            yaml_path="data/KU-PET-CT/data_splits.yaml", 
            fold_number=self.fold_number
        )

        print(f"Found {len(train_files)} training cases")
        print(f"Found {len(val_files)} validation cases")
        print(f"Found {len(test_files)} test cases")

        # debug transforms
        if self.use_distance_map:
            print("DISTANCE MAP Guided Training")
        else:
            print("SEGMENTATION MAP Guided Training")

        self.train_ds = CacheDataset(
            data=train_files,
            transform=self.train_transforms,
            cache_rate=self.cache_rate,
            num_workers=self.num_workers,
            copy_cache=False
        )

        self.val_ds = CacheDataset(
            data=val_files,
            transform=self.val_transforms,
            cache_rate=self.cache_rate,
            num_workers=self.num_workers,
            copy_cache=False
        )

        self.test_ds = CacheDataset(
            data=test_files,
            transform=self.val_transforms,
            cache_rate=self.cache_rate,
            num_workers=self.num_workers,
            copy_cache=False
        )

        # self.val_ds = Dataset(
        #     data=val_files,
        #     transform=self.val_transforms,
        # )

        # self.test_ds = Dataset(
        #     data=test_files,
        #     transform=self.val_transforms,
        # )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            # pin_memory=True,
            # persistent_workers=True,
            # # prefetch_factor=1
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=1,
            num_workers=self.num_workers,
            # pin_memory=True,
            # persistent_workers=True,
            # # prefetch_factor=2
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=1,
            num_workers=self.num_workers,
            # pin_memory=True,
            # persistent_workers=False,
        )