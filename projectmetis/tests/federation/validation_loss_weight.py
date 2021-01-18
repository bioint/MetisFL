from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns
import numpy as np
RV = norm()
import json

val_scores1a = [(337, 263.78265765765764), (297, 285.56744260204084), (306, 301.15738824503313), (292, 285.91227833044985), (271, 378.94140625), (295, 346.58898758561645), (313, 342.56856796116506), (292, 361.2395653114187), (297, 407.60097789115645)]
val_scores1b = [(296, 2), (337, 0.05), (297, 4), (306, 0.00003), (292, 10), (295, 0.78), (313, 300), (292, 0.02), (297, 50)]

val_scores2a = [(296, 136.0808847056314), (337, 108.68414508258259), (297, 95.78102412840136), (306, 150.2253207781457), (292, 92.97015570934256), (295, 107.66616277825342), (313, 128.89150991100323), (292, 110.87702746539793), (297, 157.25207270408163)]
val_scores2b = [(296, 0.00008), (337, 108.68414508258259), (297, 95.78102412840136), (306, 150.2253207781457), (292, 92.97015570934256), (295, 107.66616277825342), (313, 128.89150991100323), (292, 110.87702746539793), (297, 157.25207270408163)]

val_scores3a = [(295, 0.17358730263905983), (295, 0.10385560336178296), (295, 0.10414746689469848), (295, 0.0797667829957727), (295, 0.1535753223994007), (296, 0.17272940105138784), (295, 0.13835683587479264), (296, 0.1051551506380986), (295, 0.0891235887187801)]

val_scores4a = [(295, 0.20068048451044787), (295, 0.20767831149166577), (295, 0.15191450510939505), (295, 0.1247320828372485), (295, 0.20070008055804528), (296, 0.16712357973482828), (295, 0.17248219006682095), (296, 0.15094366496740347), (295, 0.16003427113572213)]

val_scores5a = [(304, 1.1439112345377604), (304, 1.1599212646484376), (292, 2.0474446035122374e-06), (304, 1.2080029296875), (292, 0.00017529094404827766), (304, 1.189762471516927), (292, 0.0003050852399383862), (304, 1.2295548502604168), (292, 4.443210295069589e-06)]
val_scores5b =  [(292, 5.55936503162846), (304, 0.0002474780132373174), (292, 6.03381896761462), (304, 4.9285687661419315e-06), (292, 5.479759981887976), (304, 1.795723413427671e-05), (292, 5.873531777141004), (304, 1.2889843589315811e-05), (292, 5.5156917374026815)]

val_scores6a = [(295, 0.006192960559505306), (295, 0.008010777708602278), (296, 0.0011543882788244775), (295, 0.031277774131461364), (295, 0.013266117605444503), (295, 0.01041792435188816), (295, 0.019623637199401855), (296, 0.04252989381653457), (295, 0.0037872497349569242)]
val_scores6b = [(295, 0.026438005983012998), (296, 0.05740629202676714), (295, 0.031225459216392203), (295, 0.05227313302967646), (295, 0.005076624759256024), (295, 0.00892809564120149), (295, 0.051060523072334184), (296, 0.030579186136812073), (295, 0.002697145693922696)]

val_scores7a = [(295, 0.009539484977722168), (295, 0.00031412609738029845), (296, 0.005088468053641986), (295, 0.008686099150409438), (295, 0.000698578122952213), (295, 0.0022549353642006445), (295, 0.003613352367322739), (296, 0.0010547983768450116), (295, 0.0007369247303433614)]
val_scores7b = [(295, 0.037663678600363536), (296, 0.04782120115521011), (295, 0.006067854084380686), (295, 0.05864550316170471), (295, 0.023510604688566025), (295, 0.011674306164049122), (295, 0.013192330321220502), (296, 0.005783533480387092), (295, 0.03150962476860987)]


def get_weighting_value_weighted_mean(vloss_scores):
	norm_factor = sum([x[0] for x in vloss_scores])
	# weighting_value = sum([x[0]*x[1] for x in vloss_scores]) / norm_factor
	weighting_value = sum([x[0]*x[0]*x[1] for x in vloss_scores]) / norm_factor
	weighting_value = 1 / weighting_value
	return weighting_value

def get_weighting_value_federation_mean(vloss_scores):
	num_learners = len(vloss_scores)
	weighting_value = num_learners / sum([x[1] for x in vloss_scores])
	return weighting_value

def load_validation_scores(filename):
	with open(filename) as fin:
		execution_res = json.load(fin)
		validation_loss_means = []
		all_validation_weights = []
		validation_weights_per_learner = defaultdict(list)
		for fed_round in execution_res:
			evaluation_requests = execution_res[fed_round]['metis_grpc_evaluator_metadata']['metis_grpc_evaluator_metadata']['evaluation_requests']
			for eval_request in evaluation_requests:
				request_timestamp = eval_request['request_unix_time']
				learner_federated_validation = eval_request['learner_federated_validation']
				learner_id = eval_request['learner_id']
				validation_weights_per_learner[learner_id].append(eval_request['learner_validation_weight'])
				learner_federated_validation = sorted(learner_federated_validation, key=lambda x: x['learner_id'])
				validation_loss_means.append([ (x['validation_dataset_size'], x['validation_loss_mean']) for x in learner_federated_validation ])
				all_validation_weights.append((request_timestamp, eval_request['learner_validation_weight']))
		return validation_loss_means, all_validation_weights, validation_weights_per_learner

validation_loss_means, all_validation_weights, validation_weights_per_learner = load_validation_scores("/Users/Dstrip/PycharmProjects/ProjectMetis/projectmetis/resources/logs/testing_producibility/mnist.classes_10.clients_5F5S.Function_AsyncValidation005WithLoss.AdamOpt.learningrate_00015.batchsize_50.synchronous_False.targetexectimemins_30.UFREQUENCY=4.skewness_1_5.run_1.json")
for idx, vs in enumerate(validation_loss_means):
	print()
	print(vs)
	print("Total Loss: ", sum([x[0]*x[1] for x in vs]))
	print("Weighted Loss: ", sum([x[0]*x[1] for x in vs]) / sum([x[0] for x in vs]), 1 / sum([x[0]*x[1] for x in vs]) / sum([x[0] for x in vs]))

	print("Computed Weighting: ", all_validation_weights[idx])
	print("Proposed Weighting: ", get_weighting_value_weighted_mean(vs))
	print("Proposed Weighting: ", get_weighting_value_federation_mean(vs))

for learner_id, validation_weights in validation_weights_per_learner.items():

	m = np.mean(validation_weights)
	std = np.std(validation_weights)
	var = np.var(validation_weights)
	pdf = RV.pdf(x=np.array(validation_weights))
	print(learner_id, m, std, var)
	validation_weights = [1/x for x in validation_weights]
	init_val = validation_weights[0]
	start_point = 1
	wsize = 3

	for idx, vw in enumerate(validation_weights[1:]):
		idx = start_point + idx
		curr = vw
		# curr = np.mean(validation_weights[idx-wsize:idx])
		prev = validation_weights[idx-1]
		# prev = np.mean(validation_weights[:idx-1])
		# prev = np.mean(validation_weights[idx-wsize+1:idx+1])
		slope = (curr - prev) / wsize
		pct_change = np.abs(curr - prev)/prev * 100
		log_change = np.log(curr/prev)
		gl_loss_pct_change = (curr / min(validation_weights[:idx+1]) -1) * 100

		if log_change > 0:
			print("SLOPE: {:.2f}, PCT_CHANGE: {:.2f}, LOG_CHANGE: {:.2f}, GL_LOSS_PCT_CHANGE: {:.2f}".format(slope, pct_change, log_change, gl_loss_pct_change))

	# alpha = m
	# for idx, vw in enumerate(validation_weights):
	# 	current_var = np.var(validation_weights[:idx+1])
	# 	prob_bound = current_var / np.power(alpha,2)
	# 	print(prob_bound)

# sns.distplot(np.array(validation_weights))

# plt.show()
# print()
# v1 = [(400,0.5), (400,0.9), (400,0.8)]
# print(sum([x[0]*x[1] for x in v1]))
# print(get_weighting_value_weighted_mean(v1))
# print(get_weighting_value_federation_mean(v1))
# print()
# v2 = [(400,0.3), (400,0.9), (400,0.8)]
# print(sum([x[0]*x[1] for x in v2]))
# print(get_weighting_value_weighted_mean(v2))
# print(get_weighting_value_federation_mean(v2))


# print(get_weighting_value(val_scores1a))
# print(get_weighting_value(val_scores1b))
# print(get_weighting_value(val_scores2a))
# print(get_weighting_value(val_scores2b))
#
# print()
# print(get_weighting_value(val_scores3a))
# print(get_weighting_value(val_scores4a))
#
# print(get_weighting_value(val_scores7a))
# print(get_weighting_value(val_scores7b))