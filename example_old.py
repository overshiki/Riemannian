import torch
from torchvision.models.vgg import vgg11_bn

vgg = vgg11_bn(pretrained=False)


class model(torch.nn.Module):
	def __init__(self, ):
		super().__init__()
		self.conv1 = torch.nn.Conv2d(1, 3, 3, stride=1, padding=1)

		# self.layer = vgg.features
		self.layer1 = torch.nn.Sequential(*[vgg.features[i] for i in range(0,4,1)])

		self.layer2 = torch.nn.Sequential(*[vgg.features[i] for i in range(4,8,1)])

		self.layer3 = torch.nn.Sequential(*[vgg.features[i] for i in range(8,14,1)])

		self.layer4 = torch.nn.Sequential(*[vgg.features[i] for i in range(15,22,1)])

		self.layer5 = torch.nn.Sequential(*[vgg.features[i] for i in range(22,29,1)])

		# embedding_dim = 10
		self.linear1 = torch.nn.Linear(512, 100)
		self.linear2 = torch.nn.Linear(100, 100)
		self.linear3 = torch.nn.Linear(100, 10)

		self.sigmoid = torch.nn.Sigmoid()
		self.relu = torch.nn.ReLU()
		self.dropout = torch.nn.Dropout(p=0.5)

	def forward(self, x):
		x = self.conv1(x)
		x = self.layer1(x)

		x = self.layer2(x)

		x = self.layer3(x)

		x = self.layer4(x)

		x = self.layer5(x)

		x = torch.squeeze(x)

		x = self.linear1(x)
		x = self.relu(x)
		x = self.dropout(x)

		x = self.linear2(x)
		x = self.relu(x)
		x = self.dropout(x)

		x = self.linear3(x)
		x = self.sigmoid(x)
		return x



def test():
	for X, Y in ld.get():
		# print(X.shape, Y.shape)
		X, Y = torch.from_numpy(X).float().cuda(), torch.from_numpy(Y).long().cuda()

		p = _model(X)

		print(p.shape)

		break



_model = model()

from dataset.loader import loader
ld = loader("/media/nvme0n1/DATA/TRAININGSETS/mnist/X.pa", "/media/nvme0n1/DATA/TRAININGSETS/mnist/Y.pa")

from fisher import fisher_information_matrix


fisher_information_matrix(ld, _model)