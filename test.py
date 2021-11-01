import paddle
import numpy as np

label = np.random.randn(1,3,258,258)
label = paddle.to_tensor(label)

label = label.astype('long')
# print(label)

from paddle.io import Dataset, RandomSampler

train_dataset_size = 100
train_ids = range(train_dataset_size)
print(train_ids)

partial_size = 40
print(len(train_ids[:partial_size]))

state = paddle.load('./resnet50_v1s-25a187fa.pdparams')
print(state)