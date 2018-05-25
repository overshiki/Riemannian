import torch


class model(torch.nn.Module):
	def __init__(self, ):
		super().__init__()
		self.conv1 = torch.nn.Conv2d(1, 3, 7, stride=1, padding=1)

		self.conv2 = torch.nn.Conv2d(3, 10, 7, stride=1, padding=1)

		self.conv3 = torch.nn.Conv2d(10, 10, 7, stride=1, padding=1)

		self.conv4 = torch.nn.Conv2d(10, 10, 7, stride=1, padding=1)


		self.pool = torch.nn.MaxPool2d(3, 1)
		self.pool2 = torch.nn.MaxPool2d(4, 1)
		# embedding_dim = 10
		self.linear = torch.nn.Linear(10, 10)

		self.sigmoid = torch.nn.Sigmoid()


	def forward(self, x):
		x = self.conv1(x)
		x = self.pool(x)

		x = self.conv2(x)
		x = self.pool(x)

		x = self.conv3(x)
		x = self.pool(x)

		x = self.conv4(x)
		x = self.pool(x)

		x = self.pool2(x).squeeze()

		x = self.sigmoid(self.linear(x))

		return x



def test_model():
	from dataset.loader import loader
	ld = loader("/media/nvme0n1/DATA/TRAININGSETS/mnist/X.pa", "/media/nvme0n1/DATA/TRAININGSETS/mnist/Y.pa")

	_model = model().cuda()
	for X, Y in ld.get():
		# print(X.shape, Y.shape)
		X, Y = torch.from_numpy(X).float().cuda(), torch.from_numpy(Y).long().cuda()

		p = _model(X)

		print(p.shape)

		break


def test_fisher():
	_model = model()

	from dataset.loader import loader
	ld = loader("/media/nvme0n1/DATA/TRAININGSETS/mnist/X.pa", "/media/nvme0n1/DATA/TRAININGSETS/mnist/Y.pa")

	from fisher import fisher_information_matrix


	fisher = fisher_information_matrix(ld, _model)

	print(fisher.shape)

if __name__ == '__main__':
	# test_model()
	test_fisher()