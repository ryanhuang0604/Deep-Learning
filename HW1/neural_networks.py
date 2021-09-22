import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import math


NO_ACTIVATE = False
IS_LINEAR = True
epoch_count = 100000
loss_threshold = 0.005
lr = 0.1


def generate_linear(n=100):
	pts = np.random.uniform(0, 1, (n, 2))
	inputs = []
	labels = []
	for pt in pts:
		inputs.append([pt[0], pt[1]])
		#distance = (pt[0] - pt[1]) / 1.414
		if pt[0] > pt[1]:
			labels.append(0)
		else:
			labels.append(1)

	return np.array(inputs), np.array(labels).reshape(n, 1)


def generate_XOR_easy():
	inputs = []
	labels = []
	for i in range(11):
		inputs.append([0.1*i, 0.1*i])
		labels.append(0)

		if 0.1*i == 0.5:
			continue

		inputs.append([0.1*i, 1-0.1*i])
		labels.append(1)

	return np.array(inputs), np.array(labels).reshape(21, 1)


'''
def show_result(x, y, pred_y):
	plt.subplot(1, 2, 1)
	plt.title("Ground truth", fontsize=18)
	for i in range(x.shape[0]):
		if y[i] == 0:
			plt.plot(x[i][0], x[i][1], 'ro')
		else:
			plt.plot(x[i][0], x[i][1], 'bo')

	plt.subplot(1, 2, 2)
	plt.title("Predict result", fontsize=18)
	for i in range(x.shape[0]):
		if pred_y[i] == 0:
			plt.plot(x[i][0], x[i][1], 'ro')
		else:
			plt.plot(x[i][0], x[i][1], 'bo')
	plt.show()
'''


def show_result(x, y, pred_y, accuracy):
	pred_y = np.round(pred_y)
	cm = LinearSegmentedColormap.from_list('mymap', [(1, 0, 0), (0, 0, 1)], N=2)
	plt.figure(figsize=(10, 5))
	plt.subplot(1, 2, 1)
	plt.title("Ground truth", fontsize=18)
	plt.scatter(x[:, 0], x[:, 1], c=y[:, 0], cmap=cm)

	plt.subplot(1, 2, 2)
	plt.title(f"Predict result (Accuracy: {accuracy:.2f}%)", fontsize=18)
	plt.scatter(x[:, 0], x[:, 1], c=pred_y[:, 0], cmap=cm)

	plt.show()


def sigmoid(x):
	return 1.0 / (1.0 + np.exp(-x))


def derivative_sigmoid(x):
	return np.multiply(x, 1.0 - x)


def calc_loss(y, y_hat):
	return np.mean((y - y_hat)**2)


def derivative_loss(y, y_hat):
	return (y - y_hat)*(2/y.shape[0])


def show_learning_curve(loss):
	plt.plot(loss)
	plt.title("Learning Curve")
	plt.xlabel("epoch")
	plt.ylabel("loss")
	plt.show()


class Layer():
	def __init__(self, input_size, output_size):
		self.w = np.random.normal(0, 1, (input_size+1, output_size))

	def forward(self, x):
		x = np.append(x, np.ones((x.shape[0], 1)), axis=1)
		#print(x.shape)		# (100, 3)
		self.forward_gradient = x
		if NO_ACTIVATE:
			self.y = x @ self.w
		else:
			self.y = sigmoid(x @ self.w)

		return self.y

	def backward(self, derivative_C):
		if NO_ACTIVATE:
			self.backward_gradient = derivative_C
		else:
			self.backward_gradient = np.multiply(derivative_sigmoid(self.y), derivative_C)

		return self.backward_gradient @ self.w[:-1].T

	def update(self, learning_rate):
		self.gradient = self.forward_gradient.T @ self.backward_gradient
		self.w -= learning_rate * self.gradient

		return self.gradient


class NeuralNetwork():
	def __init__(self, sizes, learning_rate):
		self.learning_rate = learning_rate

		size_in, size_hid_1, size_hid_2, size_out = sizes

		self.layer = []
		self.layer.append(Layer(size_in, size_hid_1))
		self.layer.append(Layer(size_hid_1, size_hid_2))
		self.layer.append(Layer(size_hid_2, size_out))
		#print(self.layer)

	def forward(self, x):
		for l in self.layer:
			x = l.forward(x)
		return x

	def backward(self, grad_output):
		for l in self.layer[::-1]:
			grad_output = l.backward(grad_output)

	def update(self):
		for l in self.layer:
			gradients = l.update(self.learning_rate)


if __name__ == '__main__':
	nn = NeuralNetwork([2, 4, 4, 1], learning_rate=lr)
	if IS_LINEAR:
		x, y = generate_linear()
	else:	# XOR
		x, y = generate_XOR_easy()

	is_stop = False
	loss_list = []
	for epoch in range(epoch_count):
		if not is_stop:
			y_pred = nn.forward(x)
			loss = calc_loss(y_pred, y)
			nn.backward(derivative_loss(y_pred, y))
			nn.update()
			loss_list.append(loss)

			if loss < loss_threshold:
				print(f"Converge in {epoch} epoch!")
				is_stop = True
		
		'''		
		if math.isnan(loss):
			print("nan epoch: ", epoch)
			break
		'''

		if epoch % 500 == 0 or is_stop:
			print(f"epoch: {epoch}\tloss: {loss}")

		if is_stop:
			break

	# Show result
	y_pred = nn.forward(x)
	accuracy = np.count_nonzero(np.round(y_pred) == y) * 100 / len(y_pred)
	show_result(x, y, y_pred, accuracy)
	print("======================================")
	print("Test loss:\t", calc_loss(y_pred, y))
	print(f"Test accuracy:\t {accuracy}%")
	print("prediction:\n", y_pred)
	print("======================================")
	show_learning_curve(loss_list)