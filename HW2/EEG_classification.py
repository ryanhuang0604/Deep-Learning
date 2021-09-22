from functools import reduce
import dataloader
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def gen_dataset(train_x, train_y, test_x, test_y):
	datasets = []
	for x, y in [(train_x, train_y), (test_x, test_y)]:
		x = torch.stack([torch.Tensor(x[i]) for i in range(x.shape[0])])
		y = torch.stack([torch.Tensor(y[i:i+1]) for i in range(y.shape[0])])
		datasets += [TensorDataset(x, y)]
		
	return datasets


train_dataset, test_dataset = gen_dataset(*dataloader.read_bci_data())


class EEGNet(nn.Module):
	def __init__(self, activation=None, dropout=0.25):
		super(EEGNet, self).__init__()

		if not activation:
			activation = nn.ELU

		self.firstconv = nn.Sequential(
			nn.Conv2d(1, 16, kernel_size=(1, 51), stride=(1, 1), padding=(0, 25), bias=False),
			nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True)
		)
		self.depthwiseConv = nn.Sequential(
			nn.Conv2d(16, 32, kernel_size=(2, 1), stride=(1, 1), groups=16, bias=False),
			nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True),
			activation(),
			nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=0),
			nn.Dropout(p=dropout)
		)
		self.separableConv = nn.Sequential(
			nn.Conv2d(32, 32, kernel_size=(1, 15), stride=(1, 1), padding=(0, 7), bias=False),
			nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True),
			activation(),
			nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8), padding=0),
			nn.Dropout(p=dropout)
		)
		self.classify = nn.Sequential(
			nn.Linear(736, 2, bias=True)
		)


	def forward(self, x):
		x = self.firstconv(x)
		x = self.depthwiseConv(x)
		x = self.separableConv(x)
		# flatten
		x = x.view(-1, self.classify[0].in_features)
		x = self.classify(x)

		return x


class DeepConvNet(nn.Module):
	def __init__(self, activation=None, deepconv=[25, 50, 100, 200], dropout=0.5):
		super(DeepConvNet, self).__init__()

		if not activation:
			activation = nn.ELU

		self.deepconv = deepconv
		self.conv0 = nn.Sequential(
			nn.Conv2d(1, deepconv[0], kernel_size=(1, 5), stride=(1, 1), padding=(0, 0), bias=True),
			nn.Conv2d(deepconv[0], deepconv[0], kernel_size=(2, 1), stride=(1, 1), padding=(0, 0), bias=True),
			nn.BatchNorm2d(deepconv[0]),
			activation(),
			nn.MaxPool2d(kernel_size=(1, 2)),
			nn.Dropout(p=dropout)
		)

		for idx in range(1, len(deepconv)):
			setattr(self, 'conv'+str(idx), nn.Sequential(
				nn.Conv2d(deepconv[idx-1], deepconv[idx], kernel_size=(1, 5), stride=(1, 1), padding=(0, 0), bias=True),
				nn.BatchNorm2d(deepconv[idx]),
				activation(),
				nn.MaxPool2d(kernel_size=(1, 2)),
				nn.Dropout(p=dropout)
			))

		flatten_size = deepconv[-1] * reduce(lambda x, _: round((x-4)/2), deepconv, 750)
		self.classify = nn.Sequential(
			nn.Linear(flatten_size, 2, bias=True),
		)


	def forward(self, x):
		for i in range(len(self.deepconv)):
			x = getattr(self, 'conv'+str(i))(x)
		# flatten
		x = x.view(-1, self.classify[0].in_features)
		x = self.classify(x)

		return x


class AccuracyResult():
	def __init__(self):
		self.df = pd.DataFrame(columns=["ReLU", "Leaky ReLU", "ELU"])
		self.mapping = {
			'ELU': 'elu',
			'ReLU': 'relu',
			'Leaky ReLU': 'leaky_relu',
		}


	def add(self, modelName, Accs, para):
		rows = [0.0]*len(self.df.columns)
		if modelName in self.df.index:
			rows = self.df.loc[modelName]
		for idx, col in enumerate(self.df.columns):
			if Accs[self.mapping[col] + '_test']:
				acc = max(Accs[self.mapping[col] + '_test'])
				if acc > rows[idx]:
					rows[idx] = acc

		if len(rows) != len(self.df.columns):
			raise AttributeError("Not enougth columns")
			
		self.df.loc[modelName] = rows

		# plot
		fig = plt.figure(figsize=(8, 4.5))
		plt.title("Activation function comparison (" + modelName + ")")
		plt.xlabel("Epoch")
		plt.ylabel("Accuracy(%)")

		for label, data in Accs.items():
			plt.plot(range(1, len(data)+1), data, '--' if 'test' in label else '-', label=label)

		plt.legend()
		
		name = "_"
		for i in para:
			name = name + i + "=" + str(para[i]) + ", "
		name = name[:-2]
		fig.savefig(fname=os.path.join('.', modelName + name + ".png"), dpi=300, bbox_inches="tight")
		#fig.savefig(fname=os.path.join('.', modelName + "_e=" + str(epoch_size) + ", b=" + str(batch_size) + ", l=" + str(learning_rate) + ".png"), dpi=300, bbox_inches="tight")


	def show(self, para):
		name = ""
		for i in para:
			name = name + i + ": " + str(para[i]) + ", "
		name = name[:-2]
		print("\n", name, "\n")
		print(self.df, "\n")


def runModels(models, epoch_size, batch_size, learning_rate, optimizer=optim.Adam, criterion=nn.CrossEntropyLoss()):
	train_loader = DataLoader(train_dataset, batch_size=batch_size)
	test_loader = DataLoader(test_dataset, len(test_dataset))

	Accs = {
		**{key+"_train": [] for key in models},
		**{key+"_test": [] for key in models}
	}

	optimizers = {
		key: optimizer(value.parameters(), lr=learning_rate)
		for key, value in models.items()
	}
	for epoch in range(epoch_size):
		train_correct = {key: 0.0 for key in models}
		test_correct = {key: 0.0 for key in models}

		# training multiple model
		for model in models.values():
			model.train()

		for idx, data in enumerate(train_loader):
			x, y = data
			inputs = x.to(device)
			labels = y.to(device).long().view(-1)

			for optimizer in optimizers.values():
				optimizer.zero_grad()

			for key, model in models.items():
				outputs = model.forward(inputs)
				loss = criterion(outputs, labels)
				loss.backward()

				train_correct[key] += (torch.max(outputs, 1)[1] == labels).sum().item()

			for optimizer in optimizers.values():
				optimizer.step()

		# testing multiple model
		for model in models.values():
			model.eval()
		with torch.no_grad():
			for _, data in enumerate(test_loader):
				x, y = data
				inputs = x.to(device)
				labels = y.to(device)

				for key, model in models.items():
					outputs = model.forward(inputs)

					test_correct[key] += (torch.max(outputs, 1)[1] == labels.long().view(-1)).sum().item()

		for key, value in train_correct.items():
			Accs[key+"_train"] += [(value*100.0) / len(train_dataset)]

		for key, value in test_correct.items():
			Accs[key+"_test"] += [(value*100.0) / len(test_dataset)]

	# epoch end
	torch.cuda.empty_cache()

	return Accs


if __name__ == '__main__':
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print("pytorch device: ", device)

	AccRes = AccuracyResult()

	# EEGNet
	print("Training & Testing EEGNet")
	models = {
		"relu": EEGNet(nn.ReLU).to(device),
		"leaky_relu": EEGNet(nn.LeakyReLU).to(device),
		"elu": EEGNet(nn.ELU).to(device),
	}
	para = {
		"epoch_size": 300,
		"batch_size": 64,
		"learning_rate": 1e-2
	}
	Accs = runModels(models, epoch_size=300, batch_size=64, learning_rate=1e-2)
	AccRes.add("EEGNet", Accs, para)

	# DeepConvNet
	print("Training & Testing DeepConvNet")
	models = {
		"relu": DeepConvNet(nn.ReLU).to(device),
		"leaky_relu": DeepConvNet(nn.LeakyReLU).to(device),
		"elu": DeepConvNet(nn.ELU).to(device),
	}
	para = {
		"epoch_size": 300,
		"batch_size": 64,
		"learning_rate": 1e-2
	}
	Accs = runModels(models, epoch_size=300, batch_size=64, learning_rate=1e-2)
	AccRes.add("DeepConvNet", Accs, para)

	AccRes.show(para)