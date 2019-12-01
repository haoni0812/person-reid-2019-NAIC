import torch
import torch.nn.functional as F
import numpy as np
import scipy.io as io
from tools import CatMeter, cosine_dist, euclidean_dist, re_ranking


def generate_jsonfile(distmat, dataset, topk):
	"""
	Args:
		distmat (numpy.ndarray): distance matrix of shape (num_query, num_gallery).
		dataset (tuple): a 2-tuple containing (query, gallery), each of which contains
			tuples of (img_path(s), pid, camid).
	"""

	num_q, num_g = distmat.shape
	indices = np.argsort(distmat, axis=1)

	query, gallery = dataset
	assert num_q == len(query)
	assert num_g == len(gallery)

	print('Compute result with top-{} ranks'.format(topk))
	print('# query: {}\n# gallery {}'.format(num_q, num_g))

	result_dict = {}
	qlist = []

	for q_idx in range(num_q):
		qimg_path, qpid = query[q_idx]
		query_name = qimg_path.replace('/home/kangning/Competition1/query_a/', '')

		g_num = 0
		glist = []
		glist.append(query_name)
		for g_idx in indices[q_idx, :]:
			gimg_path, gpid = gallery[g_idx]
			gallery_name = gimg_path.replace('/home/kangning/Competition1/gallery_a/', '')

			if g_num < topk:
				glist.append(gallery_name)
			g_num += 1
		qlist.append(glist)

	for i in range(len(qlist)):
		for j in range(1, len(qlist[i])):
			result_dict.setdefault(qlist[i][0], []).append(qlist[i][j])

	# generate json
	import json

	json = json.dumps(result_dict)
	jsonfile = 'result_4768_one_ibn101a_SA_rerank+4768_one_ibn101a_rankedloss_rerank+4768_one_dense161_cbloss_beta_rerank0.8.json'

	with open(jsonfile, 'w') as f:
		f.write(json)

	print("Successfully generate jsonfile: {}".format(jsonfile))


def evaluate(config, base, loaders, dataset):

	base.set_eval()
	_loaders = []

	# meters
	query_features_meter, query_pids_meter, query_cids_meter = CatMeter(), CatMeter(), CatMeter()
	gallery_features_meter, gallery_pids_meter, gallery_cids_meter = CatMeter(), CatMeter(), CatMeter()

	# init dataset
	if dataset == 'comp_official':
		_datasets = [loaders.comp_official_query_samples.samples, loaders.comp_official_gallery_samples.samples]
		_loaders = [loaders.comp_official_query_loader, loaders.comp_official_gallery_loader]
	elif dataset == 'comp_val':
		_datasets = [loaders.comp_val_query_samples.samples, loaders.comp_val_gallery_samples.samples]
		_loaders = [loaders.comp_val_query_loader, loaders.comp_val_gallery_loader]

	# compute query and gallery features
	with torch.no_grad():
		for loader_id, loader in enumerate(_loaders):
			for data in loader:
				# compute feautres
				images, pids = data
				features = base.model(images)
				# features, _ = base.model(images)
				# save as query features
				if loader_id == 0:
					query_features_meter.update(features.data)
					query_pids_meter.update(pids)
				# save as gallery features
				elif loader_id == 1:
					gallery_features_meter.update(features.data)
					gallery_pids_meter.update(pids)

	# get torch.Tensor
	query_features = query_features_meter.get_val()
	gallery_features = gallery_features_meter.get_val()

	# query_features = F.tanh(query_features)
	# gallery_features = F.tanh(gallery_features)


	# io.savemat('4768_offi_base_densenet161_augmentation.mat', {'query': query_features.cpu().numpy(), 'gallery': gallery_features.cpu().numpy()})

	# m1 = io.loadmat("4768_one_ibn101a_SA.mat")
	# m2 = io.loadmat("4768_one_base_ibn50a.mat")
	#
	# query_features = torch.from_numpy(m1['query'] + m2['query']*10)
	# gallery_features = torch.from_numpy(m1['gallery'] + m2['gallery']*10)

	# distance = -cosine_dist(query_features, gallery_features).data.cpu().numpy()
	# distance = euclidean_dist(query_features, gallery_features).data.cpu().numpy()
	# distance = re_ranking(query_features, gallery_features)

	d1 = np.load("/home/kangning/AI-baseline-master/distance_4768_one_ibn101a_SA_rerank.npy")
	# d2 = np.load("/home/kangning/AI-baseline-master/distance_4768_one_ibn101a_cbloss_beta_rerank.npy")
	d2 = np.load("/home/kangning/AI-baseline-master/distance_4768_one_ibn101a_rankedloss_rerank.npy")
	# d3 = np.load("/home/kangning/AI-baseline-master/distance_4768_one_base_ibn50a_rerank.npy")
	# d3 = np.load("/home/kangning/AI-baseline-master/distance_4768_one_base_densenet161_rerank.npy")
	d3 = np.load("/home/kangning/AI-baseline-master/distance_4768_one_dense161_cbloss_beta_rerank.npy")
	# d4 = np.load("/home/kangning/AI-baseline-master/distance_4768_offi_base_ibn101a_woRE_rerank.npy")

	# d2 = np.load("/home/kangning/AI-baseline-master/dist_mgn_nh.npy")
	# distance = 0.3 * d1 + d2
	distance = d1 + d2 + 0.8 * d3

	generate_jsonfile(distance, _datasets, 200)


	# n1 = 0
	# # for i in range(len(d1)):
	# for j in range(200):
	# 	if d1[10][j] < 0:
	# 		n1 += 1
	# print(n1)
	#
	# n2 = 0
	# # for i in range(len(d2)):
	# for j in range(200):
	# 	if d2[10][j] < 0:
	# 		n2 += 1
	# print(n2)
	#
	# n = 0
	# # for i in range(len(distance)):
	# for j in range(200):
	# 	if distance[10][j] < 0:
	# 		n += 1
	# print(n)



