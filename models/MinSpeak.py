from torch import nn
from models.backbone import *


#Create the twin network with the backbone model and the projection head
class MinSpeak(nn.Module):

	def __init__(self, args):
		self.args = args
		self.backbone = ResidualBLSTM(Resblock, [2])
		self.end_dim = self.backbone.flatten.end_dim
		self.dim_out = 567
		#projection head
		self.proj_head = nn.Sequential(
			nn.Linear(self.end_dim , self.end_dim , bias=False),
			nn.BatchNorm1d(self.end_dim ),
			nn.ReLU(inplace=True),
			nn.Linear(self.end_dim , self.end_dim , bias=False),
			nn.BatchNorm1d(self.end_dim ),
			nn.ReLU(inplace=True),
			nn.BatchNorm1d(self.dim_out, affine=False)
		)
		self.predictor = nn.Sequential(
			nn.Linear(self.dim_out, self.end_dim, bias=False),
			nn.BatchNorm1d(self.end_dim),
			nn.ReLU(inplace=True),  # hidden layer
			nn.Linear(self.end_dim, self.dim_out)
		)

	def forward(self, x1, x2):
		#Input of positive speaker samples x1 and x2

		o1 = self.proj_head(self.backbone(x1))
		o2 = self.proj_head(self.backbone(x2))

		p1 = self.predictor(o1)
		p2 = self.predictor(o2)

		return p1, p2, o1.detach(), o2.detach()