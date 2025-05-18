import os
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

def download():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        # www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        # updated by Xu for new dataset link;
        # the file is from: https://huggingface.co/datasets/Msun/modelnet40/tree/main
        www = "https://github.com/ma-xu/pointMLP-pytorch/releases/download/Modenet40_dataset/modelnet40_ply_hdf5_2048.zip"
        zipfile = os.path.basename(www)
        os.system('wget %s  --no-check-certificate; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))

def load_data(partition):
    download()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5'%partition)):
        # print(f"h5_name: {h5_name}")
        f = h5py.File(h5_name,'r')
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label

def random_point_dropout(pc, max_dropout_ratio=0.875):
    ''' batch_pc: BxNx3 '''
    # for b in range(batch_pc.shape[0]):
    dropout_ratio = np.random.random()*max_dropout_ratio # 0~0.875    
    drop_idx = np.where(np.random.random((pc.shape[0]))<=dropout_ratio)[0]
    # print ('use random drop', len(drop_idx))

    if len(drop_idx)>0:
        pc[drop_idx,:] = pc[0,:] # set to the first point
    return pc

def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud

def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud

def label_to_category_name(label):
    categories = [
         'airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 
         'car', 'chair', 'cone', 'cup', 'curtain', 'desk', 'door',
         'dresser', 'flower_pot', 'glass_box', 'guitar', 'keyboard', 
         'lamp', 'laptop', 'mantel', 'monitor',
         'night_stand', 'person', 'piano', 'plant', 
         'radio', 'range_hood', 'sink', 'sofa', 
         'stairs', 'stool', 'table', 'tent', 
         'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox'
         ]
    return categories[label]

class UniformSampling:
    def __init__(self, num_points, dropout_rate=1.0, select_first=False, seed=None):
        self.num_points = num_points
        self.dropout_rate = dropout_rate
        self.select_first = select_first
        self.seed = seed

    def __call__(self, points):
        N = points.shape[0]
        rng = np.random.default_rng(seed=self.seed)
        if self.num_points > 0:
            if self.select_first and N >= self.num_points:
                choice = np.arange(self.num_points)
            else:
                choice = rng.choice(N, self.num_points, replace=(N < self.num_points))
        else:
            choice = rng.choice(N, int(N * self.dropout_rate), replace=False)
        return points[choice]
    
    def __repr__(self):
        return f"{self.__class__.__name__}(num_points={self.num_points}, dropout_rate={self.dropout_rate}, select_first={self.select_first}, seed={self.seed})"

class SplitSampling:
    def __init__(self, num_points, low_prob=0.25, seed=None):
        self.seed = seed
        self.num_points = num_points
        self.low_prob = low_prob

    def __call__(self, points):
        N = points.shape[0]
        coord_min = np.min(points, axis=0)
        coord_max = np.max(points, axis=0)
        axis = np.argmax(coord_max - coord_min)
        selected = []
        rng = np.random.default_rng(seed=self.seed)

        while len(selected) < self.num_points:
            for i in range(N):
                pos = (points[i, axis] - coord_min[axis]) / (coord_max[axis] - coord_min[axis])
                prob = 1.0 if pos > 0.5 else self.low_prob
                if rng.random() < prob:
                    selected.append(points[i])
                if len(selected) >= self.num_points:
                    break

        return np.array(selected)

    def __repr__(self):
        return f"{self.__class__.__name__}(num_points={self.num_points}, low_prob={self.low_prob}, seed={self.seed})"

class GradientSampling:
    def __init__(self, num_points, seed=None):
        self.seed = seed
        self.num_points = num_points

    def __call__(self, points):
        N = points.shape[0]
        coord_min = np.min(points, axis=0)
        coord_max = np.max(points, axis=0)
        axis = np.argmax(coord_max - coord_min)
        selected = []
        rng = np.random.default_rng(seed=self.seed)

        while len(selected) < self.num_points:
            for i in range(N):
                val = (points[i, axis] - coord_min[axis] - 0.2 * (coord_max[axis] - coord_min[axis])) / (
                        0.6 * (coord_max[axis] - coord_min[axis]))
                prob = np.clip(val, 0.01, 1.0) ** 0.5
                if rng.random() < prob:
                    selected.append(points[i])
                if len(selected) >= self.num_points:
                    break

        return np.array(selected)

    def __repr__(self):
        return f"{self.__class__.__name__}(num_points={self.num_points}, seed={self.seed})"

class ModelNet40(Dataset):
    def __init__(self, partition='train', transform=None):
        self.data, self.label = load_data(partition)
        self.partition = partition        
        self.transform = transform
        
        # Apply the transform to the data if provided
        if self.transform:
            self.data = [self.transform(data) for data in self.data]
        else:
            self.data = [data for data in self.data]
        self.data = np.array(self.data)
        print("#" * 20 + f" {partition} set " + "#" * 20, flush=True)
        print(f"data shape: {self.data.shape}", flush=True)
        print(f"label shape: {self.label.shape}", flush=True)
        print(f'transform: {self.transform}', flush=True)
        print("#" * 20 + " end " + "#" * 20, flush=True)

    def __getitem__(self, item):
        pointcloud = self.data[item]
        label = self.label[item]
        if self.partition == 'train':
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    # Choose a sampling strategy
    transform = UniformSampling(num_points=1024, dropout_rate=1.0)
    # transform = SplitSampling(num_points=1024)
    # transform = GradientSampling(num_points=1024)

    # Create dataset with the sampling transform
    train = ModelNet40(transform=transform)
    test = ModelNet40('test', transform=transform)
    # for data, label in train:
    #     print(data.shape)
    #     print(label.shape)
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train, num_workers=4,
                              batch_size=32, shuffle=True, drop_last=True)
    for batch_idx, (data, label) in enumerate(train_loader):
        print(f"batch_idx: {batch_idx}  | data shape: {data.shape} | ;lable shape: {label.shape}")

    train_set = ModelNet40(partition='train')
    test_set = ModelNet40(partition='test')
    print(f"train_set size {train_set.__len__()}")
    print(f"test_set size {test_set.__len__()}")
