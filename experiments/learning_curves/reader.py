import pprint, pickle

pkl_file = open('/home/zimoqingfeng/rlSource/maddpg/experiments/learning_curves/001_rewards.pkl', 'rb')

data1 = pickle.load(pkl_file)
pprint.pprint(data1)

# data2 = pickle.load(pkl_file)
# pprint.pprint(data2)

pkl_file.close()