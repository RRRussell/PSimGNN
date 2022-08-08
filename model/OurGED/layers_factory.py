from config import FLAGS
from layers import NodeEmbedding, NodeEmbeddingCombinator, \
	NodeEmbeddingInteraction, Sequence, Fancy, Loss, MatchingMatrixComp
# from layers_gmn_icml import GMNPropagator, GMNAggregator, GMNLoss, MLP
# from layers_simgnn import ANPM
from layers_subgraph import ANPM_FAST, GMNPropagator_FAST, GMNAggregator_FAST, GMNLoss_FAST, MLP_FAST
# from layers_gsim_cnn import GraphConvolutionCollector, CNN
# from layers_mat_loss import OurLossFunction
import torch.nn as nn


def create_layers(model, pattern, num_layers):
	layers = nn.ModuleList()
	for i in range(1, num_layers + 1):  # 1-indexed
		sp = vars(FLAGS)['{}_{}'.format(pattern, i)].split(':')
		name = sp[0]
		layer_info = {}
		if len(sp) > 1:
			assert (len(sp) == 2)
			for spec in sp[1].split(','):
				ssp = spec.split('=')
				layer_info[ssp[0]] = '='.join(ssp[1:])  # could have '=' in layer_info
		if name in layer_ctors:
			layers.append(layer_ctors[name](layer_info, model, i, layers))
		else:
			raise ValueError('Unknown layer {}'.format(name))
	return layers


def create_NodeEmbedding_layer(lf, model, layer_id, *unused):
	_check_spec([4, 5], lf, 'NodeEmbedding')
	input_dim = lf.get('input_dim')
	if not input_dim:
		if layer_id != 1:
			raise RuntimeError(
				'The input dim for layer must be specified'.format(layer_id))
		input_dim = model.num_node_feat
	else:
		input_dim = int(input_dim)
	return NodeEmbedding(
		type=lf['type'],
		in_dim=input_dim,
		out_dim=int(lf['output_dim']),
		act=lf['act'],
		bn=_parse_as_bool(lf['bn']))


def create_NodeEmbeddingInteraction_layer(lf, *unused):
	_check_spec([2], lf, 'NodeEmbeddingInteraction')
	return NodeEmbeddingInteraction(
		type=lf['type'],
		in_dim=int(lf['input_dim']))


def create_NodeEmbeddingCombinator_layer(lf, model, layer_id, layers):
	_check_spec([3], lf, 'NodeEmbeddingCombinator')
	return NodeEmbeddingCombinator(
		dim_size=int(lf['dim_size']),
		from_layers=_parse_as_int_list(lf['from_layers']),
		layers=layers,
		num_node_feat=model.num_node_feat,
		style=lf['style'])


def create_Sequence_layer(lf, *unused):
	_check_spec([2], lf, 'Sequence')
	return Sequence(
		type=lf['type'],
		in_dim=int(lf['input_dim']))


def create_Fancy_layer(lf, *unused):
	_check_spec([1], lf, 'Fancy')
	return Fancy(in_dim=int(lf['input_dim']))


def create_Loss_layer(lf, *unused):
	_check_spec([1], lf, 'Loss')
	return Loss(
		type=lf['type'])


def create_GMNEncoder_layer(lf, model, layer_id, *unused):
	_check_spec([2, 3, 4], lf, 'GMNEncoder')
	if layer_id == 1:
		input_dim = model.num_node_feat
	else:
		input_dim = int(lf['input_dim'])
	return MLP(
		input_dim=input_dim,
		output_dim=int(lf['output_dim']),
		activation_type=lf['act']
	)


def create_GMNPropagator_layer(lf, model, layer_id, *unused):
	_check_spec([3, 4, 5], lf, 'GMNPropagator')

	f_node = lf.get('f_node')
	if not f_node:
		f_node = 'MLP'
	return GMNPropagator(
		input_dim=int(lf['input_dim']),
		output_dim=int(lf['output_dim']),
		f_node=f_node,
		more_nn=lf['more_nn']
	)


def create_GMNAggregator_layer(lf, *unused):
	_check_spec([2], lf, 'GMNAggregator')
	return GMNAggregator(
		input_dim=int(lf['input_dim']),
		output_dim=int(lf['output_dim'])
	)


def create_GMNLoss_layer(lf, *unused):
	_check_spec([0, 1], lf, 'GMNLoss')
	return GMNLoss(ds_metric=lf['ds_metric'])

def create_GMNEncoder_layer_FAST(lf, model, layer_id, *unused):
	_check_spec([2, 3, 4], lf, 'GMNEncoder')
	if layer_id == 1:
		input_dim = model.num_node_feat
	else:
		input_dim = int(lf['input_dim'])
	return MLP_FAST(
		input_dim=input_dim,
		output_dim=int(lf['output_dim']),
		activation_type=lf['act']
	)


def create_GMNPropagator_layer_FAST(lf, model, layer_id, *unused):
	_check_spec([3, 4, 5], lf, 'GMNPropagator')

	f_node = lf.get('f_node')
	if not f_node:
		f_node = 'MLP'
	return GMNPropagator_FAST(
		input_dim=int(lf['input_dim']),
		output_dim=int(lf['output_dim']),
		f_node=f_node,
		more_nn=lf['more_nn']
	)


def create_GMNAggregator_layer_FAST(lf, *unused):
	_check_spec([2], lf, 'GMNAggregator')
	return GMNAggregator_FAST(
		input_dim=int(lf['input_dim']),
		output_dim=int(lf['output_dim'])
	)


def create_GMNLoss_layer_FAST(lf, *unused):
	_check_spec([0, 1], lf, 'GMNLoss')
	return GMNLoss_FAST(ds_metric=lf['ds_metric'])


def create_ANPM_layer(lf, *unused):
	_check_spec([14], lf, 'ANPM')
	return ANPM(
		input_dim=int(lf['input_dim']),
		att_times=int(lf['att_times']),
		att_num=int(lf['att_num']),
		att_style=lf['att_style'],
		att_weight=_parse_as_bool(lf['att_weight']),
		feature_map_dim=int(lf['feature_map_dim']),
		bias=_parse_as_bool(lf['bias']),
		ntn_inneract=lf['ntn_inneract'],
		apply_u=_parse_as_bool(lf['apply_u']),
		mne_inneract=lf['mne_inneract'],
		mne_method=lf['mne_method'],
		branch_style=lf['branch_style'],
		reduce_factor=int(lf['reduce_factor']),
		criterion=lf['criterion'])

def create_ANPM_FAST_layer(lf, *unused):
	_check_spec([4], lf, 'ANPM_FAST')
	return ANPM_FAST(
		input_dim=int(lf['input_dim']),
		feature_map_dim=int(lf['feature_map_dim']),
		reduce_factor=int(lf['reduce_factor']),
		criterion=lf['criterion'])


def create_GraphConvolutionCollector_layer(lf, *unused):
	_check_spec([5], lf, 'GraphConvolutionCollector')
	return GraphConvolutionCollector(
		gcn_num=int(lf['gcn_num']),
		fix_size=int(lf['fix_size']),
		mode=int(lf["mode"]),
		padding_value=int(lf["padding_value"]),
		align_corners=_parse_as_bool(lf["align_corners"])
	)


def create_CNN_layer(lf, *unused):
	_check_spec([9], lf, 'CNN')
	return CNN(
		in_channels=int(lf['in_channels']),
		out_channels=int(lf['out_channels']),
		kernel_size=int(lf["kernel_size"]),
		stride=int(lf["stride"]),
		gcn_num=int(lf["gcn_num"]),
		bias=_parse_as_bool(lf['bias']),
		poolsize=int(lf["poolsize"]),
		act=lf['act'],
		end_cnn=_parse_as_bool(lf['end_cnn'])
	)


def create_MLP_layer(lf, *unused):
	_check_spec([2, 3, 4, 5], lf, 'MLP')
	return MLP(
		input_dim=int(lf['input_dim']),
		output_dim=int(lf['output_dim']),
		activation_type=lf['act'],
		num_hidden_lyr=int(lf['num_hidden_lyr']),
		hidden_channels=_parse_as_int_list(lf["hidden_channels"])
	)


def create_PCA_layer(lf, model, *unused):
	_check_spec([6], lf, 'MLP')
	return PCAModel(desc=[model.num_node_feat] + _parse_as_int_list(lf['desc']),
					tao1=float(lf['tao1']),
					tao2=float(lf['tao2']),
					sinkhorn_iters=int(lf['sinkhorn_iters']),
					sinkhorn_eps=float(lf['sinkhorn_eps']),
					aff_max=int(lf['aff_max'])
					)


def create_OurLossFunction(lf, model, *unused):
	return OurLossFunction(n_features=model.num_node_feat,
						   alpha=float(lf['alpha']),
						   beta=float(lf['beta']),
						   gamma=float(lf['gamma']),
						   tau=float(lf['tau']),
						   y_from=lf['y_from'],
						   z_from=lf['z_from']
						   )


def create_MatchingMatrixComp(lf, *unused):
	return MatchingMatrixComp(opt=lf['opt'])

"""
Register the constructor caller function here.
"""
layer_ctors = {
	'NodeEmbedding': create_NodeEmbedding_layer,
	'NodeEmbeddingCombinator': create_NodeEmbeddingCombinator_layer,
	'NodeEmbeddingInteraction': create_NodeEmbeddingInteraction_layer,
	'Loss': create_Loss_layer,
	'Sequence': create_Sequence_layer,
	'Fancy': create_Fancy_layer,
	'GMNEncoder': create_GMNEncoder_layer,
	'GMNPropagator': create_GMNPropagator_layer,
	'GMNAggregator': create_GMNAggregator_layer,
	'GMNLoss': create_GMNLoss_layer,
	'GMNEncoder_FAST': create_GMNEncoder_layer_FAST,
	'GMNPropagator_FAST': create_GMNPropagator_layer_FAST,
	'GMNAggregator_FAST': create_GMNAggregator_layer_FAST,
	'GMNLoss_FAST': create_GMNLoss_layer_FAST,
	'simgnn': create_ANPM_layer,
	'simgnn_fast': create_ANPM_FAST_layer,
	'GraphConvolutionCollector': create_GraphConvolutionCollector_layer,
	'CNN': create_CNN_layer,
	'MLP': create_MLP_layer,
	'OurLossFunction': create_OurLossFunction,
	'MatchingMatrixComp': create_MatchingMatrixComp
}


def _check_spec(allowed_nums, lf, ln):
	if len(lf) not in allowed_nums:
		raise ValueError('{} layer must have {} specs NOT {} {}'.
						 format(ln, allowed_nums, len(lf), lf))


def _parse_as_bool(b):
	if b == 'True':
		return True
	elif b == 'False':
		return False
	else:
		raise RuntimeError('Unknown bool string {}'.format(b))


def _parse_as_int_list(il):
	rtn = []
	for x in il.split('_'):
		x = int(x)
		rtn.append(x)
	return rtn