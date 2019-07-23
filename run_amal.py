import sys
sys.path.append('c:\\projects\\SKVP-Python')
import skvp
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts'))
from sequence_segmentor import find_optimal_segmentation
from create_textual_feedback import generate_feedback_text
import create_textual_feedback
import numpy as np
import math
import scipy.stats

def list_of_vectors_of_concatenated_active_joint_gradients(vid, active_joints):
		grad = skvp.gradient(vid)
		lst = []
		for frame in grad.frames:
			vec = []
			for j in active_joints:
				vec.extend(frame[j])
			lst.append(np.array(vec))
		
		return lst

def detect_active_joints(vid):
	joint_vars = []
	for joint_id in range(vid.get_num_joints()):
		x_locations = [frame[joint_id][0] for frame in vid.frames]
		y_locations = [frame[joint_id][1] for frame in vid.frames]
		z_locations = [frame[joint_id][2] for frame in vid.frames]
		var_x = np.var(x_locations)
		var_y = np.var(y_locations)
		var_z = np.var(z_locations)
		joint_vars.append(np.linalg.norm([var_x, var_y, var_z]))
	thresh = np.mean(joint_vars)
	
	
	return [i for i in range(len(joint_vars)) if joint_vars[i] > thresh]



def detect_vids_active_joints(vids):
	active_joints_counter = {}
	for i, vid in enumerate(vids):
		active_joints = detect_active_joints(vid)
		for aj in active_joints:
			if aj not in active_joints_counter:
				active_joints_counter[aj] = 0
			active_joints_counter[aj] += 1
	thresh = int(round(len(vids) * 0.3))
	
	return sorted([aj for aj in active_joints_counter if active_joints_counter[aj] > thresh])

def detect_rest_sequences(vid, active_joints, exponent = -1.5):
	der = skvp.median(skvp.gradient(vid))
	rest_index_counter = {}
	for joint in active_joints:
		stats = [np.linalg.norm(f[joint]) ** exponent for f in der.frames]
		stats_mean = np.mean(stats)
		rest_indices = [i for i, val in enumerate(stats) if val > stats_mean]
		margin = int(round(len(der) * 0.075))
		rest_indices = [val for val in rest_indices if val > margin and val < len(der) - margin]
		for ri in rest_indices:
			if ri not in rest_index_counter:
				rest_index_counter[ri] = 0
			rest_index_counter[ri] += 1
	thresh = int(len(active_joints) * 0.3)
	rest_indices = sorted([ri for ri in rest_index_counter if rest_index_counter[ri] > thresh])
	sequences = []
	first = None
	last = None
	for ind in rest_indices:
		if first == None:
			first = ind
			last = ind
			continue
		if ind - last > len(der) * 0.1:
			sequences.append((first, last))
			first = ind
		last = ind
	if last != None:
		sequences.append((first, last))
	
	index_trans_func = lambda x : x + 1
	
	return sequences, index_trans_func

def warp_video(vid, orig_indices, target_indices):
	new_vid = vid[0:0]
	orig_prev = 0
	target_prev = 0
	for orig, target in zip(orig_indices, target_indices):
		orig = int(orig)
		target = int(target)
		new_vid += skvp.create_length_scaled_video(vid[orig_prev:orig], num_frames = target - target_prev)
		orig_prev = orig
		target_prev = target
	new_vid += skvp.create_length_scaled_video(vid[orig_prev:], num_frames = len(vid) - target_prev)	

	return new_vid

def parse_model_file(filepath):
	f = open(filepath, 'r')
	lines = [l.strip() for l in f.readlines() if l.strip() != '']
	f.close()
	started_stats = False
	active_joints = None
	vid_length = None
	num_rests = None
	warp_indices = None
	stats = []
	for line in lines:
		if line == '--stats--':
			started_stats = True
			continue
		if started_stats:
			stat_dict = eval(line)
			stats.append(stat_dict)
			continue
		if line.startswith('ActiveJoints='):
			active_joints = eval(line.split('=', 1)[1])
			continue
		if line.startswith('VidLength='):
			vid_length = eval(line.split('=', 1)[1])
			continue
		if line.startswith('NumRests='):
			num_rests = eval(line.split('=', 1)[1])
			continue
		if line.startswith('WarpIndices='):
			warp_indices = eval(line.split('=', 1)[1])
			continue
		if line.startswith('ConnectionLengths='):
			connection_lengths = eval(line.split('=', 1)[1])
			continue
			
	return vid_length, active_joints, num_rests, warp_indices, connection_lengths, stats

vector_score = lambda x : np.linalg.norm(x) / float(len(x) ** 0.5)

def weighted_vector_score(vec, weights):
	myvec = np.array(vec)
	weighted_vec = myvec * weights
	
	return np.linalg.norm(weighted_vec) / np.linalg.norm(weights)

def produce_feedback(costed_stats, group_weights, ablation):
	def stat_to_param_type(stat):
		if 'joint' in stat:
			return '{0:d},{1}'.format(stat['joint'], stat['type'])
		if 'joint_i' in stat:
			return '{0:d},{1:d},{2}'.format(stat['joint_i'], stat['joint_j'], stat['type'])
		if 'joint_trio' in stat:
			return '{0},{1}'.format(str(stat['joint_trio']), stat['type'])
		return None
	stats_by_type = {}
	for stat in costed_stats:
		if 'type_group' not in stat or (stat['type_group'] != 'ActiveJoint' and stat['type_group'] != 'NonActiveJoint'):
			continue
		key = stat_to_param_type(stat)
		if key == None:
			continue
		if key not in stats_by_type:
			stats_by_type[key] = []
		stats_by_type[key].append(stat)
	feedbacks = []
	num_total_stats = 0
	num_good_stats = 0
	cost_sum = 0
	positive_cost_sum = 0
	active_bad_segment_lengths = []
	active_bad_segment_nums = []
	nonactive_bad_segment_nums = []
	for key, stats in stats_by_type.items():
		stats.sort(key = lambda x : x['frame'])
		type_group = stats[0]['type_group']
		num_total_stats += len(stats)
		seq = ['0' if stat['cost'] < 2.5 else ('+' if stat['test_val'] > stat['mean'] else '-') for stat in stats]
		seg = find_optimal_segmentation(seq)
		num_active_bad = 0
		num_nonactive_bad = 0
		for part in seg:
			if part['classification'] == 'zero':
				for i in range(part['start'], part['end'] + 1):
					if'segmentation' not in ablation:
						stats[i]['cost'] = -3
				num_good_stats += (part['end'] - part['start'] + 1)
				continue
			if type_group == 'ActiveJoint':
				num_active_bad += 1
				active_bad_segment_lengths.append(part['end'] - part['start'] + 1)
			elif type_group == 'NonActiveJoint':
				num_nonactive_bad += 1
			fb = {'key' : key, 'num_key_stats' : len(stats), 'seq' : part, 'cost' : sum([stats[i]['cost'] for i in range(part['start'], part['end'] + 1)]), 'type_group' : type_group}
			feedbacks.append(fb)
		if num_active_bad > 0:
			active_bad_segment_nums.append(num_active_bad)
		if num_nonactive_bad > 0:
			nonactive_bad_segment_nums.append(num_nonactive_bad)
		for s in stats:
			cost_sum += s['cost']
			if s['cost'] > 0:
				positive_cost_sum += s['cost']
	
	avg_active_bad_segment_length = 0 if len(active_bad_segment_lengths) == 0 else (sum(active_bad_segment_lengths) / float(len(active_bad_segment_lengths)))
	avg_param_num_active_segments = 0 if len(active_bad_segment_nums) == 0 else (sum(active_bad_segment_nums) / float(len(active_bad_segment_nums)))
	avg_param_num_nonactive_segments = 0 if len(nonactive_bad_segment_nums) == 0 else (sum(nonactive_bad_segment_nums) / float(len(nonactive_bad_segment_nums)))
	
	sequence_stats = [stat for stat in costed_stats if stat['type'] == 'sequence_length']
	sequence_stats.sort(key = lambda x : x['start_frame'])
	curr_feedback = None
	for stat in sequence_stats:
		if stat['cost'] < 2.5:
			if curr_feedback != None:
				feedbacks.append(curr_feedback)
				curr_feedback = None
			continue
		classification = 'plus' if stat['test_val'] > stat['mean'] else 'minus'
		if curr_feedback != None:
			if classification == curr_feedback['seq']['classification']:
				curr_feedback['seq']['end'] = stat['end_frame']
				curr_feedback['cost'] += stat['cost']
				continue
			feedbacks.append(curr_feedback)
			curr_feedback = {'seq' : {'start' : stat['start_frame'], 'end' : stat['end_frame'], 'classification' : classification}, 'key' : stat['type'], 'cost' : stat['cost'], 'type_group' : stat['type_group']}
		else:
			curr_feedback = {'seq' : {'start' : stat['start_frame'], 'end' : stat['end_frame'], 'classification' : classification}, 'key' : stat['type'], 'cost' : stat['cost'], 'type_group' : stat['type_group']}
	vid_len_stats = [stat for stat in costed_stats if stat['type'] == 'video_length_frames_raw']
	if len(vid_len_stats) != 1:
		raise Exception('Number of stats of video length frames raw: {0:d}'.format(len(vid_len_stats)))
	if vid_len_stats[0]['cost'] > 2.5:
		feedbacks.append({'key' : 'video_length_frames_raw', 'cost' : vid_len_stats[0]['cost'], 'type_group' : vid_len_stats[0]['type_group'], 'seq' : {'start' : 0, 'end' : 0, 'classification' : ('minus' if vid_len_stats[0]['test_val'] < vid_len_stats[0]['mean'] else 'plus')}})
	if curr_feedback != None:
		feedbacks.append(curr_feedback)
	
	
	
	return feedbacks, avg_active_bad_segment_length, avg_param_num_active_segments, avg_param_num_nonactive_segments
	
def test(model_file, input_video_file, warp_mode, ablation, ref_vid_path):
	vid = skvp.load(input_video_file)
	vid = skvp.project_to_body_plane(vid, 0, 4, 8)
	orig_vid_len_frames = len(vid)
	vid_len, active_joints, num_rests, warp_indices, connection_lengths, stats = parse_model_file(model_file)
	vid = skvp.scaled_connections(vid, connection_lengths)
	vid = skvp.create_length_scaled_video(vid, num_frames = vid_len)
	vid = skvp.median(vid)
	pyr = skvp.pyramid(vid, [3, 3, 3], [1, 1, 1])
	vid = pyr.get_level(1)
	rest_sequences = []
	exponent = -1.5
	ignore_subsequence_lengths = False
	while len(rest_sequences) < num_rests:
		exponent *= 0.9
		if exponent > -0.5:
			rest_sequences = [(warp_indices[i - 1], warp_indices[i])  for i in range(len(warp_indices)) if i % 2 == 1]
			index_transform_function = lambda x : x + 1
			ignore_subsequence_lengths = True
			break
		rest_sequences, index_transform_function = detect_rest_sequences(vid, active_joints, exponent)
	vec = []
	for sq in rest_sequences:
		if sq[0] == sq[1]:
			sq = (sq[0] - 1, sq[1] + 1)
		vec.extend(sq)
	vec = [index_transform_function(val) for val in vec]
	if warp_mode in ('our'):
		vid = warp_video(vid, vec, warp_indices)
	elif warp_mode == 'dtw':
		refvid = skvp.load(ref_vid_path)
		ref_vals = list_of_vectors_of_concatenated_active_joint_gradients(refvid, active_joints)
		vid_vals = list_of_vectors_of_concatenated_active_joint_gradients(vid, active_joints)
		matches, cost, mapping_1, mapping_2, matrix = simpledtw.dtw(ref_vals, vid_vals)
		warped = vid[:1] # Starting from one frame, as grad has n-1 frames
		for mapped_indices in mapping_1:
			warped.add_frame(vid.frames[mapped_indices[-1]])
		vid = warped
	der = skvp.gradient(vid)
	costs = []
	group_weights = {'ActiveJoint' : 0.73, 'NonActiveJoint' : 0.02, 'Time' : 0.25}
	if 'active' in ablation:
		group_weights = {'ActiveJoint' : 0.15, 'NonActiveJoint' : 0.6, 'Time' : 0.25}
	if 'time' in ablation:
		group_weights['ActiveJoint'] /= (1.0 - group_weights['Time'])
		group_weights['NonActiveJoint'] /= (1.0 - group_weights['Time'])
		group_weights['Time'] = 0
	for stat in stats:
		if stat['type'] == 'sequence_length':
			stat['start_frame'] = (vec[stat['sequence_num'] - 1] + 1) if stat['sequence_num'] > 0 else 0
			stat['end_frame'] = vec[stat['sequence_num']] if stat['sequence_num'] < len(vec) else len(vid) - 1
		if ignore_subsequence_lengths and stat['type'] == 'sequence_length':
			print('Ignoring seq lengths')
			continue
		if 'joint' in stat or 'joint_trio' in stat:
			stat['type_group'] = 'NonActiveJoint'
		elif 'joint_i' in stat:
			stat['type_group'] = 'NoGroupIgnoreMe' if 'active' not in ablation else 'NonActiveJoint'
		else:
			stat['type_group'] = 'Time'
		if 'joint' in stat and stat['joint'] in active_joints:
			stat['type_group'] = 'ActiveJoint'
		if 'joint_i' in stat and stat['joint_i'] in active_joints and 'joint_j' in stat and stat['joint_j'] in active_joints:
			stat['type_group'] = 'ActiveJoint'
		if 'joint_trio' in stat:
			for j in stat['joint_trio']:
				if j in active_joints:
					stat['type_group'] = 'ActiveJoint'
		if stat['type'] == 'location_x':
			val = vid.frames[stat['frame']][stat['joint']][0]
		elif stat['type'] == 'location_y':
			val = vid.frames[stat['frame']][stat['joint']][1]
		elif stat['type'] == 'location_z':
			val = vid.frames[stat['frame']][stat['joint']][2]
		elif stat['type'] == 'gradient_x':
			val = der.frames[stat['frame']][stat['joint']][0]
		elif stat['type'] == 'gradient_y':
			val = der.frames[stat['frame']][stat['joint']][1]
		elif stat['type'] == 'gradient_z':
			val = der.frames[stat['frame']][stat['joint']][2]
		elif stat['type'] == 'joint_distance':
			val = np.linalg.norm(vid.frames[stat['frame']][stat['joint_i']] - vid.frames[stat['frame']][stat['joint_j']])
		elif stat['type'] == 'joint_angles':
			joint_trio = stat['joint_trio']
			vec_1 = vid.frames[stat['frame']][joint_trio[0]] - vid.frames[stat['frame']][joint_trio[1]]
			vec_2 = vid.frames[stat['frame']][joint_trio[2]] - vid.frames[stat['frame']][joint_trio[1]]
			val = np.arccos(vec_1.dot(vec_2) / (np.linalg.norm(vec_1) * np.linalg.norm(vec_2)))
		elif stat['type'] == 'video_length_frames_raw':
			val = orig_vid_len_frames
		elif stat['type'] == 'sequence_length':
			if stat['sequence_num'] == 0:
				val = vec[0] + 1
			elif stat['sequence_num'] < len(vec):
				val = vec[stat['sequence_num']] - vec[stat['sequence_num'] - 1] + 1
			else:
				val = len(vid) - vec[stat['sequence_num'] - 1] + 1
		else:
			continue
		# Avoiding division by 0 when we normalize the distance
		val = round(val, 7)
		stat['mean'] = round(stat['mean'], 7)
		stat['dist_std'] = round(stat['dist_std'], 7)
		stat['dist_mean'] = round(stat['dist_mean'], 7)
		try:
			dist = abs(val - stat['mean'])
		except:
			print('stat is: {0}'.format(str(stat)))
			print('val is: {0}'.format(str(val)))
			print('mean is: {0}'.format(str(stat['mean'])))
		if stat['dist_std'] == 0:
			if dist - stat['dist_mean'] == 0:
				dist_in_stds = 0
			else:
				dist_in_stds = np.inf
		else:
			dist_in_stds = (dist - stat['dist_mean']) / stat['dist_std']
		stat['cost'] = dist_in_stds
		stat['test_val'] = val
	costed_stats = [s for s in stats if 'cost' in s]# and s['type'] == 'joint_angles']
	costed_active_joint_stats = [s for s in costed_stats if s['type_group'] == 'ActiveJoint']
	costed_nonactive_joint_stats = [s for s in costed_stats if s['type_group'] == 'NonActiveJoint']
	costed_time_stats = [s for s in costed_stats if s['type_group'] == 'Time']
	lambdaa = 2
	group_to_stats = {'Time' : costed_time_stats, 'ActiveJoint' : costed_active_joint_stats, 'NonActiveJoint' : costed_nonactive_joint_stats}
	
	feedback_items, avg_active_bad_segment_length, avg_param_num_active_segments, avg_param_num_nonactive_segments = produce_feedback(costed_stats, group_weights, ablation)
	
	scorelambda = 2.9
	scorelambdatime = 7.5
	
	
	
	for item in feedback_items:
		# Here we duplicate the weight manipulation, just to be able to sort the feedbacks
		# The same manupulation will be applied on real items (depending on ablation settings) - we don't apply ablation on feedback
		item['cost'] *= group_weights[item['type_group']]
		item['cost'] /= (scorelambdatime if item['type_group'] == 'Time' else scorelambda)
		item['cost'] /= len([s['cost'] for s in group_to_stats[item['type_group']]])
		if item['type_group'] == 'ActiveJoint':
			item['cost'] *= (0.75 ** avg_param_num_active_segments)
	feedback_items.sort(key = lambda x : x['cost'], reverse = True)
	
	if len(feedback_items) > 5:
		feedback_items = feedback_items[:5] 
	last_cost = None
	for i, fi in enumerate(feedback_items):
		if last_cost == None or last_cost < 2 * fi['cost']:
			last_cost = fi['cost']
			continue
		# Removing irrelevant feedback
		feedback_items = feedback_items[:i]
		break
	feedback_items = [generate_feedback_text(fi, 15) for fi in feedback_items]
	if len(feedback_items) == 0:
		print('FEEDBACK: empty! congrats :)')
	for feedback_item in feedback_items:
		print('FEEDBACK: ' + str(feedback_item))
		
	if 'diminish' not in ablation and 'active' not in ablation:
		for s in costed_active_joint_stats:
			s['cost'] *= (0.75 ** avg_param_num_active_segments)
			
	#Vector scores
	active_joint_score = max(0, 1.0 - abs(sum(s['cost'] for s in costed_active_joint_stats if s['cost'] > 0) / (scorelambda * len([s['cost'] for s in costed_active_joint_stats]))))
	nonactive_joint_score = max(0, 1.0 - abs(sum(s['cost'] for s in costed_nonactive_joint_stats if s['cost'] > 0) / (scorelambda * len([s['cost'] for s in costed_nonactive_joint_stats]))))
	time_score = max(0, 1.0 - abs(sum(s['cost'] for s in costed_time_stats if s['cost'] > 2.5) / (scorelambdatime * len([s['cost'] for s in costed_time_stats]))))
	
	print('Scores: Active: {0:f}, NonActive: {1:f}, Time: {2:f}'.format(active_joint_score, nonactive_joint_score, time_score))
	final_score = active_joint_score * group_weights['ActiveJoint'] + nonactive_joint_score * group_weights['NonActiveJoint'] + time_score * group_weights['Time']
	
	print('Score: {0:f}'.format(final_score))

def fixed_std(vals):
	var = np.var(vals) * len(vals) / float(len(vals) - 1.0)
	
	return var ** 0.5
	
def get_centroid_dist_mean_and_dist_std(list_of_vectors):
	centroid = np.mean(list_of_vectors, axis = 0)
	dists = []
	for i, vec in enumerate(list_of_vectors):
		all_except_i = list_of_vectors[:i] + list_of_vectors[i+1:]
		tmp_centroid = np.mean(all_except_i, axis = 0)
		dist = np.linalg.norm(vec - tmp_centroid)
		dists.append(dist)
	dist_mean = np.mean(dists)
	dist_std = fixed_std(dists)

	return centroid, dist_mean, dist_std
			
def get_joint_angle_vectors(active_joints, connections):
	connections = [(a -1, b - 1) for (a, b) in connections]
	relevant_connections = [conn for conn in connections if conn[0] in active_joints or conn[1] in active_joints]
	relevant_joints = {conn[0] for conn in relevant_connections} | {conn[1] for conn in relevant_connections}
	joint_angles = []
	for joint in relevant_joints:
		edges = [tuple(sorted(edge)) for edge in connections if edge[0] == joint or edge[1] == joint]
		if len(edges) == 1:
			continue
		for i, edge in enumerate(edges):
			other_joint1 = edge[0] if edge[1] == joint else edge[1]
			for edge2 in edges[i+1:]:
				other_joint2 = edge2[0] if edge2[1] == joint else edge2[1]
				angle = {'mid' : joint, 'ends': set((other_joint1, other_joint2))}
				if angle not in joint_angles:
					joint_angles.append(angle)

	return joint_angles

def find_all_joint_trios(connections):
	# Will return a list of joint trios: (lower_index, join_index, higher_index)
	# For example, if joint 1 is connected with 5 and joint 5 is connected with 3, then the tuple (1,5,3) will be included in the returned list
	joint_to_neighbors = {}
	for joint_1, joint_2 in connections:
		joint_1 -= 1
		joint_2 -= 1
		if joint_1 not in joint_to_neighbors:
			joint_to_neighbors[joint_1] = []
		joint_to_neighbors[joint_1].append(joint_2)
		if joint_2 not in joint_to_neighbors:
			joint_to_neighbors[joint_2] = []
		joint_to_neighbors[joint_2].append(joint_1)
	joint_trios = []
	for middle_joint, neighbors in joint_to_neighbors.items():
		if len(neighbors) < 2:
			continue
		for i, neighbor_1 in enumerate(neighbors):
			for neighbor_2 in neighbors[i + 1:]:
				if neighbor_1 > neighbor_2:
					joint_trios.append((neighbor_2, middle_joint, neighbor_1))
				else:
					joint_trios.append((neighbor_1, middle_joint, neighbor_2))
					
	return joint_trios

def match_sequences(a, b):
	matches = []
	for i, seq_i in enumerate(a):
		for j, seq_j in enumerate(b):
			matches.append({'match' : (i,j), 'distance' : abs(seq_i[0] - seq_j[0]) + abs(seq_i[1] - seq_j[1])})
	matches.sort(key = lambda x : x['distance'])
	chosen_is = set()
	chosen_js = set()
	selected_matches = []
	for match in matches:
		if match['match'][0] not in chosen_is and match['match'][1] not in chosen_js:
			chosen_is.add(match['match'][0])
			chosen_js.add(match['match'][1])
			selected_matches.append(match['match'])	
	
	return selected_matches

def our_warping_function(normalized_vids_with_same_length):
	vids = normalized_vids_with_same_length
	print('Detecting active joints...')
	active_joints = detect_vids_active_joints(vids)
	print('Detecting rest sequences...')
	rest_sequence_results = [detect_rest_sequences(vid, active_joints) for vid in vids]
	rest_sequences = [res[0] for res in rest_sequence_results]
	index_transform_functions = [res[1] for res in rest_sequence_results]
	num_rests_in_motion = int(round(np.median([len(r) for r in rest_sequences])))
	print('Number of motion\'s rests: {0:d}'.format(num_rests_in_motion))
	print('Detecting looser rests in videos with less detected rests...')
	exponents = []
	for i, vid in enumerate(vids):
		exponent = -1.5
		while len(rest_sequences[i]) < num_rests_in_motion:
			exponent *= 0.9
			rest_sequences[i], index_transform_functions[i] = detect_rest_sequences(vid, active_joints, exponent)
		exponents.append(exponent)
	print('Computing average rest sequence indices, over all videos with correct number of sequences')
	average_rest_sequences = []
	for r in range(num_rests_in_motion):
		sum_left = 0.0
		sum_right = 0.0
		num = 0
		for rs, vid in zip(rest_sequences, vids):
			if len(rs) != num_rests_in_motion:
				continue
			num += 1
			sum_left += rs[r][0]
			sum_right += rs[r][1]
		avg_left = sum_left / float(num)
		avg_right = sum_right / float(num)
		average_rest_sequences.append((avg_left, avg_right))
	print('For videos with too many rest sequences, leaving only the ones that are closest to the average')
	for i, (rs, vid) in enumerate(zip(rest_sequences, vids)):
		if len(rs) == num_rests_in_motion:
			continue
		print('Doing it for video number: {0:d}'.format(i))
		matches = match_sequences(average_rest_sequences, rs)
		surviving_rest_sequence_indices = set((m[1] for m in matches))
		for si in range(len(rs) - 1, -1, -1):
			if si not in surviving_rest_sequence_indices:
				del rs[si]
	print('Choosing reference video...')
	rest_sequences_as_vectors = []
	for i, rs in enumerate(rest_sequences):
		vec = []
		for sq in rs:
			if sq[0] == sq[1]:
				sq = (sq[0] - 1, sq[1] + 1)
			vec.extend(sq)
		vec = [index_transform_functions[i](val) for val in vec]
		rest_sequences_as_vectors.append(np.array(vec))
	centroid = np.mean(rest_sequences_as_vectors, axis = 0)
	dists_from_centroid = [np.linalg.norm(centroid - vec) for vec in rest_sequences_as_vectors]
	ref_vid_index = np.argmin(dists_from_centroid)
	print('Reference video index: {0:d}'.format(ref_vid_index))
	print('Warping other training videos to match the rests of the reference videos')
	warped_vids = [warp_video(vid, rest_sequences_as_vectors[i], rest_sequences_as_vectors[ref_vid_index]) for i, vid in enumerate(vids)]
	
	
	return ref_vid_index, warped_vids, active_joints, num_rests_in_motion, list(rest_sequences_as_vectors[ref_vid_index]), rest_sequences_as_vectors

def get_connection_mean_lengths(vids):
	edge_lengths_per_vid = [skvp.connection_lengths(vid) for vid in vids]
	lengths_per_edge = {}
	for edge_lenghts in edge_lengths_per_vid:
		for edge, length in edge_lenghts.items():
			if edge not in lengths_per_edge:
				lengths_per_edge[edge] = []
			lengths_per_edge[edge].append(length)
	for edge, lengths in lengths_per_edge.items():
		if len(lengths) != len(vids):
			raise Exception('Length of edge {0} is only available in {1:d} of {2:d} videos'.format(str(edge), len(lenghts), len(vids)))
	
	return {edge : np.mean(lengths) for edge, lengths in lengths_per_edge.items()}
	
def warp_using_dtw(vids, ref_vid_index, active_joints):			
	vid_lists_of_vectors = [list_of_vectors_of_concatenated_active_joint_gradients(vid, active_joints) for vid in vids]
	warped_vids = []
	for i, vid in enumerate(vids):
		if i == ref_vid_index:
			warped_vids.append(vid)
			continue
		matches, cost, mapping_1, mapping_2, matrix = simpledtw.dtw(vid_lists_of_vectors[ref_vid_index], vid_lists_of_vectors[i])
		warped = vid[:1] # Starting from one frame, as grad has n-1 frames
		for mapped_indices in mapping_1:
			warped.add_frame(vid.frames[mapped_indices[-1]])
		warped_vids.append(warped)
	
	return warped_vids
	
def train(input_dir, output_file, warp_mode, output_ref_vid):
	vid_files = [os.path.join(input_dir, fname) for fname in os.listdir(input_dir)]
	print('Loading training videos...')
	vids = [skvp.load(path) for path in vid_files]
	print('Normalizing training videos...')
	vids = [skvp.project_to_body_plane(vid, 0, 4, 8) for vid in vids]
	connections_mean_lengths = get_connection_mean_lengths(vids)
	vids = [skvp.scaled_connections(vid, connections_mean_lengths) for vid in vids]
	vids_original_lengths_before_filters = [len(vid) for vid in vids]
	vids_mean_len_before_filters = int(round(np.mean(vids_original_lengths_before_filters)))	
	print('Scaling training videos to have the same length')
	vids = [skvp.create_length_scaled_video(vid, num_frames = vids_mean_len_before_filters) for vid in vids]
	print('Applying filters')
	vids = [skvp.median(vid) for vid in vids]
	vid_pyramids = [skvp.pyramid(vid, [3, 3, 3], [1, 1, 1]) for vid in vids]
	vids = [pyr.get_level(1) for pyr in vid_pyramids]
	vids_new_len = len(vids[0])
	# Saving original videos before tempora alignment
	vids_nowrap = [vid[:] for vid in vids]
	print('Applying temporal alignment')
	ref_vid_index, vids, active_joints, num_rests_in_motion, ref_vid_warping_indices, all_vids_warping_sequences = our_warping_function(vids)
	if warp_mode == 'none':
		vids = vids_nowrap
	if warp_mode == 'dtw':
		vids = warp_using_dtw(vids_nowrap, ref_vid_index, active_joints)
	if output_ref_vid != None:
		skvp.dump(vids_nowrap[ref_vid_index], output_ref_vid)
	# Computing discrete temporal gradients
	ders = [skvp.gradient(vid) for vid in vids]
	connections = skvp.distinct_connections(vids[0])
	# Writing motion metadata into output model file
	f = open(output_file, 'w')
	f.write('ActiveJoints={0}\n'.format(str(active_joints)))
	f.write('VidLength={0:d}\n'.format(vids_mean_len_before_filters))
	f.write('NumRests={0:d}\n'.format(num_rests_in_motion))
	f.write('WarpIndices={0}\n'.format(ref_vid_warping_indices))
	f.write('ConnectionLengths={0}\n'.format(str(connections_mean_lengths)))
	f.write('--stats--\n')
	print('Calculating stats and writing to output file...')
	# Computing and writing time-related statistics - original video lengths and aligned sequence original lengths
	mean, dist_mean, dist_std = get_centroid_dist_mean_and_dist_std(vids_original_lengths_before_filters)
	data_unit = {'mean' : mean, 'dist_mean' : dist_mean, 'dist_std' : dist_std, 'type' : 'video_length_frames_raw'}
	f.write('{0}\n'.format(str(data_unit)))
	for i in range(len(all_vids_warping_sequences[0]) + 1):
		if i == 0:
			seq_lengths = [v[0] + 1 for v in all_vids_warping_sequences]
		elif i < len(all_vids_warping_sequences[0]):
			seq_lengths = [v[i] - v[i - 1] + 1 for v in all_vids_warping_sequences]
		else:
			seq_lengths = [vids_new_len - v[i - 1] + 1 for v in all_vids_warping_sequences]
		mean, dist_mean, dist_std = get_centroid_dist_mean_and_dist_std(seq_lengths)
		data_unit = {'mean' : mean, 'dist_mean' : dist_mean, 'dist_std' : dist_std, 'type' : 'sequence_length', 'sequence_num' : i}
		f.write('{0}\n'.format(str(data_unit)))
	# Computing and writing joint-related statistics
	for frame_num in range(vids_new_len):
		for joint in range(vids[0].get_num_joints()):
			points = [vid.frames[frame_num][joint] for vid in vids]
			xs = [p[0] for p in points]
			ys = [p[1] for p in points]
			zs = [p[2] for p in points]			
			mean, dist_mean, dist_std = get_centroid_dist_mean_and_dist_std(xs)
			data_unit = {'frame' : frame_num, 'joint' : joint, 'mean' : mean, 'dist_mean' : dist_mean, 'dist_std' : dist_std, 'type' : 'location_x'}
			f.write('{0}\n'.format(str(data_unit)))
			mean, dist_mean, dist_std = get_centroid_dist_mean_and_dist_std(ys)
			data_unit = {'frame' : frame_num, 'joint' : joint, 'mean' : mean, 'dist_mean' : dist_mean, 'dist_std' : dist_std, 'type' : 'location_y'}
			f.write('{0}\n'.format(str(data_unit)))
			mean, dist_mean, dist_std = get_centroid_dist_mean_and_dist_std(zs)
			data_unit = {'frame' : frame_num, 'joint' : joint, 'mean' : mean, 'dist_mean' : dist_mean, 'dist_std' : dist_std, 'type' : 'location_z'}
			f.write('{0}\n'.format(str(data_unit)))
			if frame_num < vids_new_len - 1: # We have one frame less in the gradient
				der_points = [vid.frames[frame_num][joint] for vid in ders]
				xs = [p[0] for p in der_points]
				ys = [p[1] for p in der_points]
				zs = [p[2] for p in der_points]			
				mean, dist_mean, dist_std = get_centroid_dist_mean_and_dist_std(xs)
				data_unit = {'frame' : frame_num, 'joint' : joint, 'mean' : mean, 'dist_mean' : dist_mean, 'dist_std' : dist_std, 'type' : 'gradient_x'}
				f.write('{0}\n'.format(str(data_unit)))
				mean, dist_mean, dist_std = get_centroid_dist_mean_and_dist_std(ys)
				data_unit = {'frame' : frame_num, 'joint' : joint, 'mean' : mean, 'dist_mean' : dist_mean, 'dist_std' : dist_std, 'type' : 'gradient_y'}
				f.write('{0}\n'.format(str(data_unit)))
				mean, dist_mean, dist_std = get_centroid_dist_mean_and_dist_std(zs)
				data_unit = {'frame' : frame_num, 'joint' : joint, 'mean' : mean, 'dist_mean' : dist_mean, 'dist_std' : dist_std, 'type' : 'gradient_z'}
				f.write('{0}\n'.format(str(data_unit)))
		for joint_i in range(vids[0].get_num_joints()):
			for joint_j in range(joint_i + 1, vids[0].get_num_joints()):
				if (joint_i, joint_j) in connections:
					# Skipping neighbors!!
					continue
				dists = [np.linalg.norm(vid.frames[frame_num][joint_i] - vid.frames[frame_num][joint_j]) for vid in vids]
				centroid, dist_mean, dist_std = get_centroid_dist_mean_and_dist_std(dists)
				data_unit = {'frame' : frame_num, 'joint_i' : joint_i, 'joint_j' : joint_j, 'mean' : float(centroid), 'dist_mean' : dist_mean, 'dist_std' : dist_std, 'type' : 'joint_distance'}
				f.write('{0}\n'.format(str(data_unit)))
		for lower_joint_index, join_index, higher_joint_index in find_all_joint_trios(vids[0].get_connections()):
			angles = []
			for vid in vids:
				vec_1 = vid.frames[frame_num][lower_joint_index] - vid.frames[frame_num][join_index]
				vec_2 = vid.frames[frame_num][higher_joint_index] - vid.frames[frame_num][join_index]
				theta = np.arccos(vec_1.dot(vec_2) / (np.linalg.norm(vec_1) * np.linalg.norm(vec_2)))
				angles.append(theta)
			centroid, dist_mean, dist_std = get_centroid_dist_mean_and_dist_std(angles)
			data_unit = {'frame' : frame_num, 'joint_trio' : [lower_joint_index, join_index, higher_joint_index], 'mean' : float(centroid), 'dist_mean' : dist_mean, 'dist_std' : dist_std, 'type' : 'joint_angles'}
			f.write('{0}\n'.format(str(data_unit)))
	f.close()

if __name__ == '__main__':
	mode = sys.argv[1]
	input_dir_or_model_file = sys.argv[2]
	output_model_file_or_input_video_file = sys.argv[3]
	warp_mode = sys.argv[4] if len(sys.argv) > 4 else 'our'
	ablation = set(sys.argv[5].split(',')) if len(sys.argv) > 5 else {'none'}
	output_or_input_ref_vid = sys.argv[6] if len(sys.argv) > 6 else None


	warp_mode = warp_mode.lower()
	if warp_mode not in ('none', 'dtw', 'our'):
		raise Exception('Warp mode must be one of {none, dtw, our}')

	if ablation != {'none'} and mode.lower() != 'test':
		raise Exception('Ablation can only be specified in test mode')
	# "No-warp" ablation is defined by "warp mode"
	for ab in ablation:
		if ab not in ('none', 'active', 'segmentation', 'time', 'diminish'):
			raise Exception('Ablation must be one of {none, active, segmentation, time, diminish}')
	if warp_mode == 'dtw':
		import simpledtw

	

	if mode.lower() == 'train':
		input_dir = input_dir_or_model_file
		output_file = output_model_file_or_input_video_file
		train(input_dir, output_file, warp_mode, output_or_input_ref_vid)
	elif mode.lower() == 'test':
		model_file = input_dir_or_model_file
		input_video_file = output_model_file_or_input_video_file
		test(model_file, input_video_file, warp_mode, ablation, output_or_input_ref_vid)
	else:
		sys.stderr.write('Illegal mode: {0}. Supported modes are "train" or "test"'.format(mode))
		sys.exit(1)

