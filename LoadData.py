
import numpy as np


# def load_next_batch(batch=100, loaded=0):
#     arrays = []
#     data = []
#     labels = []
#     apple_data = np.load('apple.npy', 'r')
#     for i in range(batch//2):
#         arrays.append([apple_data[i+loaded], [1, 0]])
#     banana_data = np.load('banana.npy', 'r')
#     for i in range(batch//2):
#         arrays.append([banana_data[i+loaded], [0, 1]])
#     ret_array = np.asarray(arrays)
#     np.random.shuffle(ret_array)
#     for k in ret_array:
#         data.append(k[0])
#         labels.append(k[1])
#     return np.asarray(data), np.asarray(labels)


def load_next_batch(batch=100, loaded=0):
    arrays = []
    data = []
    labels = []
    apple_data = np.load('apple.npy', 'r')
    for i in range(batch):
        arrays.append([apple_data[i+loaded], [1, 0]])
    banana_data = np.load('banana.npy', 'r')
    for i in range(batch//2):
        arrays.append([banana_data[i+loaded], [0, 1]])
    ret_array = np.asarray(arrays)
    np.random.shuffle(ret_array)
    for k in ret_array:
        data.append(k[0])
        labels.append(k[1])
    return np.asarray(data), np.asarray(labels)

def create_data():
    num_of_imgs = 50000
    arrays = []
    apple_data = np.load('apple.npy', 'r')
    for i in range(num_of_imgs):
        arrays.append([apple_data[i + num_of_imgs], [1, 0]])
    banana_data = np.load('banana.npy', 'r')
    for i in range(num_of_imgs):
        arrays.append([banana_data[i + num_of_imgs], [0, 1]])
    ret_array = np.asarray(arrays)
    np.random.shuffle(ret_array)
    np.save('train_data.txt', ret_array)


#d, l = load_next_batch(1000, 10000)
#for i in range(100):
#    print(l[i])