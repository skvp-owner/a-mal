import sys
import numpy as np
import math

def mean_of_consecutive_indices(first, last):
	# They are consecutive... The mean of each pair of (i, n - i - 1) gives the same whole sequence mean
	return (first + last) / 2.0

def var_of_consecutive_indices(first, last):
	mean = mean_of_consecutive_indices(first, last)
	sum_of_squared_distances = sum([(v - mean) ** 2 for v in range(first, last + 1)])
	
	return sum_of_squared_distances / (last - first) # dividing by n - 1
	
def general_var(numbers):
	mean = sum(numbers) / float(len(numbers))
	sum_of_squared_distances = sum([(v - mean) ** 2 for v in numbers])
	
	return sum_of_squared_distances / (len(numbers) - 1)

def compute_rates_and_var_scores(seq):
	desired_var = var_of_consecutive_indices(0, len(seq) - 1) if len(seq) > 1 else None
	indices_plus = []
	indices_minus = []
	indices_zero = []
	for i, val in enumerate(seq):
		if val == '+':
			indices_plus.append(i)
		elif val == '-':
			indices_minus.append(i)
		elif val == '0':
			indices_zero.append(i)
		else:
			raise Exception('Illegal char in sequence: {0}'.format(val))
	
	var_plus = general_var(indices_plus) if len(indices_plus) > 1 else 0
	var_minus = general_var(indices_minus) if len(indices_minus) > 1 else 0
	var_zero = general_var(indices_zero) if len(indices_zero) > 1 else 0
	
	return 	(
			len(indices_plus) / float(len(seq)), 
			len(indices_minus) / float(len(seq)), 
			len(indices_zero) / float(len(seq)),
			max(0, (1.0 - (((var_plus - desired_var) / desired_var) ** 2))) if desired_var != None else 1.0,
			max(0, (1.0 - (((var_minus - desired_var) / desired_var) ** 2))) if desired_var != None else 1.0,
			max(0, (1.0 - (((var_zero - desired_var) / desired_var) ** 2))) if desired_var != None else 1.0
			)
	

def classify_sequence(seq, reward_length = False):
	
	rate_plus, rate_minus, rate_zero, var_score_plus, var_score_minus, var_score_zero = compute_rates_and_var_scores(seq)
	rate_mult = min(1, min(rate_minus, rate_plus) * 2)
	var_score_mult = (var_score_minus + var_score_plus) / 2

	score_plus = rate_plus * var_score_plus / 4
	score_minus = rate_minus * var_score_minus / 4
	score_zero = rate_zero * var_score_zero / 4
	score_mult = rate_mult * var_score_mult
	score_trash = min(1.0 - var_score_plus, 1.0 - var_score_minus, 1.0 - var_score_zero)
	scores = np.array((score_plus, score_minus, score_zero, score_mult, score_trash))
	if np.max(scores) == 0:
		return 'plus', 0.0 # class name is not relevant, as score is 0. Doing it since can't normalize zero vector
	scores /= np.linalg.norm(scores, ord = 2)
	
	classes = ['plus', 'minus', 'zero', 'mult', 'trash']
	winner_class_index = np.argmax(scores)
	
	return classes[winner_class_index], (scores[winner_class_index] * ((1.0 + math.log(len(seq))) ** 0.2 if reward_length else 1.0)) if winner_class_index < 4 else 0
	
	
if __name__ == '__main__':
	sequences = [
		'+++++++++',
		'++++-++++',
		'+++++0+++',
		'---------',
		'-----+---',
		'----0-----',
		'00000000000',
		'0000000+000',
		'0000-000000',
		'0000-000+000',
		'+0-0+0-0+0-0+0',
		'+-+-+-+-+-+-+-',
		'+0-0+0-0+-+-+-',
		'+',
		'++',
		'+-',
		'+-+',
		'+--+',
		'++++++++++-----------'
	]
	for seq in sequences:
		c, score = classify_sequence(seq)
		print('Sequence: {0}. Class: {1}. Score: {2:.2f}'.format(seq, c, score))
