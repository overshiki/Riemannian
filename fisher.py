import torch


def crossEntropyElementwise(X, Y):
	X = torch.gather(X, 1, Y.unsqueeze(dim=1))
	X = torch.log(X)
	return X


def fisher_information_matrix(loader, model, device=0):
	loss = crossEntropyElementwise

	if device is not None:
		model.cuda(device)

	param_dim = 0
	for name, params in model.named_parameters():
		param_dim = param_dim+params.numel()

	print(param_dim)

	grad_total = torch.zeros(param_dim, param_dim).float()
	count = 0
	for index, (X, Y) in enumerate(loader.get()):
		X, Y = torch.from_numpy(X).float(), torch.from_numpy(Y).long()
		if device is not None:
			X = X.cuda(device)
			Y = Y.cuda(device)

		R = model(X.requires_grad_())
		L = loss(R, Y)

		param_list = []
		for name, params in model.named_parameters():
			param_list.append(params)

		for l in L:
			grad = torch.autograd.grad(l, param_list, retain_graph=True)
			grad = list(map(lambda x: x.view(x.numel()), grad))
			grad = torch.cat(grad, dim=0)

			grad_0 = torch.unsqueeze(grad, dim=0).cpu()
			grad_1 = torch.unsqueeze(grad, dim=1).cpu()
			grad = grad_0*grad_1
			grad_total = grad_total+grad
			count = count+1

	grad_mean = grad_total*1./count
	return grad_mean

