


from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
import numpy as np

class Nfold:

	def __init__(self, data, split, shuffle_flag):
		self.data = data
		self.shuffle_flag = shuffle_flag
		self.lent = len(data)
		self.label = []
		self.feature_vector = []
		self.splits = split
		self.indices = []
		self.train_idx = []
		self.test_idx = []
		self.train_data = []
		self.test_data = []

		for i, elem in enumerate(data):
			self.indices.append(elem['global_index'])

	def __iter__(self):
		#         skf=StratifiedKFold(n_splits=2, random_state=None, shuffle=False)

		kf = KFold(n_splits=self.splits, random_state=None, shuffle=self.shuffle_flag)
		#         skf.split
		self.list_of_folds = kf.split(self.indices)

		return self

	def __next__(self):
		self.train_idx, self.test_idx = next(self.list_of_folds)

		# print('train indices: ', self.train_idx, 'test indices: ', self.test_idx)

		self.train_data = list(
			map(lambda index: self.data.select_a_record_based_on_a_key(key='global_index', value=index),
			    self.train_idx))

		self.test_data = list(
			map(lambda index: self.data.select_a_record_based_on_a_key(key='global_index', value=index), self.test_idx))

		return self.train_data, self.test_data


class Stratified_Nfold:

	def __init__(self, data, split, shuffle_flag):
		self.data = data
		self.shuffle_flag = shuffle_flag
		self.lent = len(data)
		self.labels = []
		self.feature_vector = []
		self.splits = split
		self.indices = []
		self.train_idx = []
		self.test_idx = []
		self.train_data = []
		self.test_data = []

		for i, elem in enumerate(data):
			self.indices.append(elem['global_index'])

		for i, label in enumerate(data):
			self.labels.append(label['dbType'])

	def __iter__(self):

		skf = StratifiedKFold(n_splits=self.splits, random_state=None, shuffle=self.shuffle_flag)
		self.list_of_folds = skf.split(self.indices, self.labels)

		return self

	def __next__(self):

		self.train_idx, self.test_idx = next(self.list_of_folds)

		# print('train indices: ', self.train_idx, 'test indices: ', self.test_idx)

		self.train_data = list(
			map(lambda index: self.data.select_a_record_based_on_a_key(key='global_index', value=index),
			    self.train_idx))

		self.test_data = list(
			map(lambda index: self.data.select_a_record_based_on_a_key(key='global_index', value=index), self.test_idx))

		return self.train_data, self.test_data



def retrieve_trainXy_testXy_kfold(train_in_a_fold, test_in_a_fold):
	X_train = []
	y_train = []
	X_test = []
	y_test = []

	for train in train_in_a_fold:
		y_train += [train['dbType']]
		X_train += [train['feature_vector']]

	for test in test_in_a_fold:
		y_test += [test['dbType']]
		X_test += [test['feature_vector']]

	return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)
