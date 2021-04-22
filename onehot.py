from sklearn.preprocessing import OneHotEncoder
import numpy as np

data = np.array([0,1,2,3,4,5,6,7,8,9])
# 将标签转换为独热编码
encoder = OneHotEncoder()
data = encoder.fit_transform(data.reshape(data.shape[0], 1))
data = data.toarray().T
data = data.astype('uint8')
print(data[0])