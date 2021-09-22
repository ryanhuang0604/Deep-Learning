from dataloader import CLEVRDataset
from torch.utils.data import DataLoader
from util import get_test_conditions

print(get_test_conditions('data/test.json'))