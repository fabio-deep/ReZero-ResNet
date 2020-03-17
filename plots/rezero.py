from numpy import genfromtxt
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

''' ResNet-56 '''

train_error_52 = './epoch_error_train_52.csv'
train_error_52 = genfromtxt(train_error_52, delimiter=',')
valid_error_52 = './epoch_error_valid_52.csv'
valid_error_52 = genfromtxt(valid_error_52, delimiter=',')

train_error_53 = './epoch_error_train_53.csv'
train_error_53 = genfromtxt(train_error_53, delimiter=',')
valid_error_53 = './epoch_error_valid_53.csv'
valid_error_53 = genfromtxt(valid_error_53, delimiter=',')

# resnet56 (model 52)
# Training time: 127m 41s, 281 epochs
#
# Best [Valid] | epoch: 221 - loss: 0.3129 - acc: 0.9396
# [Test] loss 0.3042 - acc: 0.9356 - acc_topk: 0.9795

# resnet56 alpha (model 53)
# Training time: 134m 44s, 303 epochs
#
# Best [Valid] | epoch: 243 - loss: 0.2934 - acc: 0.9369
# [Test] loss 0.3062 - acc: 0.9345 - acc_topk: 0.9788

fig = plt.figure()
plt.xlabel('epoch', fontsize=14)
plt.ylabel('error (%)', fontsize=14)
plt.title('ResNet-56', fontsize=16)

plt.plot(train_error_53[1:,1], 100*train_error_53[1:,2], label='train (ReZero)', zorder=1)#, linewidth=1.5)
plt.plot(valid_error_53[1:,1], 100*valid_error_53[1:,2], label='valid (ReZero)', zorder=2)#, linewidth=1.5)
plt.plot(train_error_52[1:,1], 100*train_error_52[1:,2], 'c', label='train', zorder=3)#, linewidth=1.5)
plt.plot(valid_error_52[1:,1], 100*valid_error_52[1:,2], 'm', label='valid', zorder=4)#, linewidth=1.5)

plt.ticklabel_format(axis='y', style='sci')
plt.grid(True)
plt.legend(loc='upper right', fontsize='x-large') # upper right, lower right, lower left
# plt.ylim(5, 80)
# plt.xlim(1, 30)
# plt.savefig('resnet56_error_0_30.png', box_inches='tight')
plt.ylim(-2, 35)
plt.xlim(0, 250)
plt.savefig('resnet56_error.png', box_inches='tight')
plt.show()

train_loss_52 = './epoch_loss_train_52.csv'
train_loss_52 = genfromtxt(train_loss_52, delimiter=',')
valid_loss_52 = './epoch_loss_valid_52.csv'
valid_loss_52 = genfromtxt(valid_loss_52, delimiter=',')

train_loss_53 = './epoch_loss_train_53.csv'
train_loss_53 = genfromtxt(train_loss_53, delimiter=',')
valid_loss_53 = './epoch_loss_valid_53.csv'
valid_loss_53 = genfromtxt(valid_loss_53, delimiter=',')

fig = plt.figure()
plt.xlabel('epoch', fontsize=14)
plt.ylabel('loss', fontsize=14)
plt.title('ResNet-56', fontsize=16)

plt.plot(train_loss_53[1:,1], train_loss_53[1:,2], label='train (ReZero)', zorder=1)#, linewidth=1.5)
plt.plot(valid_loss_53[1:,1], valid_loss_53[1:,2], label='valid (ReZero)', zorder=2)#, linewidth=1.5)
plt.plot(train_loss_52[1:,1], train_loss_52[1:,2], 'c', label='train', zorder=3)#, linewidth=1.5)
plt.plot(valid_loss_52[1:,1], valid_loss_52[1:,2], 'm', label='valid', zorder=4)#, linewidth=1.5)

plt.ticklabel_format(axis='y', style='sci')
plt.grid(True)
plt.legend(loc='upper right', fontsize='x-large') # upper right, lower right, lower left
plt.ylim(.2, 2.5)
plt.xlim(1, 30)
plt.savefig('resnet56_loss_0_30.png', box_inches='tight')
# plt.ylim(-.1, 1.2)
# plt.xlim(0, 250)
# plt.savefig('resnet56_loss.png', box_inches='tight')
plt.show()

''' ResNet-20 '''

train_error_54 = './epoch_error_train_54.csv'
train_error_54 = genfromtxt(train_error_54, delimiter=',')
valid_error_54 = './epoch_error_valid_54.csv'
valid_error_54 = genfromtxt(valid_error_54, delimiter=',')

train_error_55 = './epoch_error_train_55.csv'
train_error_55 = genfromtxt(train_error_55, delimiter=',')
valid_error_55 = './epoch_error_valid_55.csv'
valid_error_55 = genfromtxt(valid_error_55, delimiter=',')

# resnet-20 alpha (model 54)
# Training time: 63m 9s, 327
#
# Best [Valid] | epoch: 267 - loss: 0.3237 - acc: 0.9256
# [Test] loss 0.3491 - acc: 0.9206 - acc_topk: 0.9723

# resnet-20 (model 55)
# Training time: 70m 3s, 398
#
# Best [Valid] | epoch: 338 - loss: 0.3026 - acc: 0.9237
# [Test] loss 0.3055 - acc: 0.9202 - acc_topk: 0.9742

fig = plt.figure()
plt.xlabel('epoch', fontsize=14)
plt.ylabel('error (%)', fontsize=14)
plt.title('ResNet-20', fontsize=16)

plt.plot(train_error_54[1:,1], 100*train_error_54[1:,2], label='train (ReZero)', zorder=1)#, linewidth=1.5)
plt.plot(valid_error_54[1:,1], 100*valid_error_54[1:,2], label='valid (ReZero)', zorder=2)#, linewidth=1.5)
plt.plot(train_error_55[1:,1], 100*train_error_55[1:,2], 'c', label='train', zorder=3)#, linewidth=1.5)
plt.plot(valid_error_55[1:,1], 100*valid_error_55[1:,2], 'm', label='valid', zorder=4)#, linewidth=1.5)

plt.ticklabel_format(axis='y', style='sci')
plt.grid(True)
plt.legend(loc='upper right', fontsize='x-large') # upper right, lower right, lower left
plt.ylim(5, 60)
plt.xlim(1, 30)
plt.savefig('resnet20_error_0_30.png', box_inches='tight')
# plt.ylim(-2, 35)
# plt.xlim(0, 300)
# plt.savefig('resnet20_error.png', box_inches='tight')
plt.show()

train_loss_54 = './epoch_loss_train_54.csv'
train_loss_54 = genfromtxt(train_loss_54, delimiter=',')
valid_loss_54 = './epoch_loss_valid_54.csv'
valid_loss_54 = genfromtxt(valid_loss_54, delimiter=',')

train_loss_55 = './epoch_loss_train_55.csv'
train_loss_55 = genfromtxt(train_loss_55, delimiter=',')
valid_loss_55 = './epoch_loss_valid_55.csv'
valid_loss_55 = genfromtxt(valid_loss_55, delimiter=',')

fig = plt.figure()
plt.xlabel('epoch', fontsize=14)
plt.ylabel('loss', fontsize=14)
plt.title('ResNet-20', fontsize=16)

plt.plot(train_loss_54[1:,1], train_loss_54[1:,2], label='train (ReZero)', zorder=1)#, linewidth=1.5)
plt.plot(valid_loss_54[1:,1], valid_loss_54[1:,2], label='valid (ReZero)', zorder=2)#, linewidth=1.5)
plt.plot(train_loss_55[1:,1], train_loss_55[1:,2], 'c', label='train', zorder=3)#, linewidth=1.5)
plt.plot(valid_loss_55[1:,1], valid_loss_55[1:,2], 'm', label='valid', zorder=4)#, linewidth=1.5)

plt.ticklabel_format(axis='y', style='sci')
plt.grid(True)
plt.legend(loc='upper right', fontsize='x-large') # upper right, lower right, lower left
plt.ylim(.2, 1.6)
plt.xlim(1, 30)
plt.savefig('resnet20_loss_0_30.png', box_inches='tight')
# plt.ylim(-.1, 1.2)
# plt.xlim(0, 300)
# plt.savefig('resnet20_loss.png', box_inches='tight')
plt.show()
