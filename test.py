import paddle
import numpy as np
import paddle.nn as nn

label = np.random.randn(1,3,258,258)
label = paddle.to_tensor(label)

label = label.astype('long')
# print(label)

# train_dataset_size = 100
# train_ids = range(train_dataset_size)
# print(train_ids)
#
# partial_size = 40
# print(len(train_ids[:partial_size]))

# state = paddle.load('./resnet50_v1s-25a187fa.pdparams')
# print(state)

# interp = nn.Upsample(size=(label[1], label[0]), mode='bilinear', align_corners=True)
# print(interp(label))

conv1 = nn.Conv2D(64,128,3)
conv2 = nn.Conv2D(128,256,3)
b = []
b.append(conv1)
b.append(conv2)
print(b)