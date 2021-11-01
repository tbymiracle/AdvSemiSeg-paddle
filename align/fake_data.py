import numpy as np
import paddle
fake_data = np.random.rand(7, 3, 256, 256).astype(np.float32)
fake_label = np.random.rand(7, 256, 256).astype(np.float32)
# print(fake_data)
print(fake_label)
np.save("fake_data.npy", fake_data)
np.save("fake_label.npy", fake_label)
# img = np.load('fake_data.npy')
# img = paddle.to_tensor(img)
# print(img.shape)
