from scipy.io import loadmat
import numpy as np

alice_channel_train = np.load('/home/cxq/code_10.23/20db/train_alice.npy')

alice_channel_test = np.load('/home/cxq/code_10.23/20db/test_alice.npy')

jack_channel_train = np.load('/home/cxq/code_10.23/20db/train_jack.npy')

jack_channel_test = np.load('//home/cxq/code_10.23/20db/test_jack.npy')

alice_channel_train = alice_channel_train*5e5

alice_channel_test = alice_channel_test*5e5

jack_channel_train = jack_channel_train*5e5

jack_channel_test = jack_channel_test*5e5

sampled_channel_VAE = np.concatenate((alice_channel_train,jack_channel_train), axis=0)

print("sampled_channel_VAE.shape",sampled_channel_VAE.shape)

np.save('/home/cxq/code_10.23/20db/sampled_channel_VAE.npy',sampled_channel_VAE)


width = alice_channel_train.shape[2]
quarter_width = width // 2
sampled_dim3 = alice_channel_train[:, :,  :quarter_width, :]


width = sampled_dim3.shape[3]
quarter_width = width // 8
alice_channel_train = sampled_dim3[:, :, :, :quarter_width]



print("Shape after sampling dim3 (1 out of every 4):", sampled_dim3.shape)
print("Shape after sampling dim4 (1 out of every 16):", alice_channel_train.shape)



width = alice_channel_test.shape[2]
quarter_width = width // 2
sampled_dim3 = alice_channel_test[:, :,  :quarter_width, :]



width = sampled_dim3.shape[3]
quarter_width = width // 8
alice_channel_test = sampled_dim3[:, :, :, :quarter_width]



print("Shape after sampling dim3 (1 out of every 4):", sampled_dim3.shape)
print("Shape after sampling dim4 (1 out of every 16):", alice_channel_test.shape)



width = jack_channel_train.shape[2]
quarter_width = width // 2
sampled_dim3 = jack_channel_train[:, :,  :quarter_width, :]



width = sampled_dim3.shape[3]
quarter_width = width // 8
jack_channel_train = sampled_dim3[:, :, :, :quarter_width]



print("Shape after sampling dim3 (1 out of every 4):", sampled_dim3.shape)
print("Shape after sampling dim4 (1 out of every 16):", jack_channel_train.shape)



width = jack_channel_test.shape[2]
quarter_width = width // 2
sampled_dim3 = jack_channel_test[:, :,  :quarter_width, :]



width = sampled_dim3.shape[3]
quarter_width = width // 8
jack_channel_test = sampled_dim3[:, :, :, :quarter_width]



print("Shape after sampling dim3 (1 out of every 4):", sampled_dim3.shape)
print("Shape after sampling dim4 (1 out of every 16):", jack_channel_test.shape)

channel_train_sampled = np.concatenate((alice_channel_train, alice_channel_train), axis=3)
print("channel_train_sampled",channel_train_sampled.shape)

channel_test_sampled = np.concatenate((alice_channel_test, alice_channel_test), axis=3)
print("channel_test_sampled",channel_test_sampled.shape)
np.save('/home/cxq/code_10.23/20db/channel_train_sampled.npy',channel_train_sampled)

np.save('/home/cxq/code_10.23/20db/channel_test_sampled.npy',channel_test_sampled)
