import torch
from numpy import prod


def get_shape_list(dataset='ffhq-256'):
	if dataset == 'ffhq-256':
		shapes = [(2,1), (4,4), (5,8), (10,16), (22,32), (14,64), (8,128), (1,256)]
		shape_list = []
		for m, res in shapes:
			shape_list = shape_list + m*[(16, res, res)]
	else:
		shape_list = None

	return shape_list


def tensor2vect(tensor, endblock=None):
	# 'tensor' must have shape [N,C,H,W]
	vect = []
	n_samples = tensor[0].shape[0]
	if endblock is None:
		endblock = len(tensor)

	for i in range(endblock):
		t = tensor[i]
		vect.append(t.reshape(n_samples,-1))

	output = torch.cat(vect, dim=1)

	return output


def vect2tensors(vect, endblock=None, dataset='ffhq-256'):
	tensors = []
	ind = 0
	n_samples = vect.shape[0]
	shape_list = get_shape_list(dataset)
	n_blocks = len(shape_list)

	if endblock is None:
		endblock = n_blocks

	for i,shape in enumerate(shape_list):
		n_elem = prod(shape)
		if i>=endblock:
			t = torch.zeros((n_samples, *shape), device=vect.device)
		else:
			t = vect[:, ind:ind+n_elem].reshape(n_samples, *shape)
		tensors.append(t)
		ind += n_elem

	return tensors
