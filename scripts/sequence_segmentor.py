import sys

import sequence_classifier2


def find_optimal_segmentation(seq):
	opt = {}
	choices = {}
	classifications = {}
	num_segments = {}
	
	class_0, score_0 = sequence_classifier2.classify_sequence(seq[0], True)
	choices[0] = None
	classifications[0] = class_0
	num_segments[0] = 1
	opt[0] = score_0
	
	for i in range(1, len(seq)):
		initial_class, initial_score = sequence_classifier2.classify_sequence(seq[:i + 1], True)
		choices[i] = None
		classifications[i] = initial_class
		num_segments[i] = 1
		opt[i] = initial_score
		for j in range(0, i):
			subs_class, subs_score = sequence_classifier2.classify_sequence(seq[j + 1:i + 1], True)
			div_segments = num_segments[j] + 1
			div_score = (subs_score + opt[j] * (div_segments - 1)) / div_segments
			if div_score > opt[i]:
				opt[i] = div_score
				classifications[i] = subs_class
				num_segments[i] = div_segments
				choices[i] = j
				
	segments = []
	index = len(seq) - 1
	while index != None:
		start_index = 0 if choices[index] == None else (choices[index] + 1)
		segments.append({'start' : start_index, 'end' : index, 'classification' : classifications[index]})
		index = choices[index]
	segments.reverse()
	
	return segments

if __name__ == '__main__':
	print(find_optimal_segmentation('+-+-+-+-+-+--------------------'))
	print(find_optimal_segmentation('0000-0000+000++++++++++++0+-++++++0++000000000000-00000000000000000+--+-0+++--+--+--+'))
	print(find_optimal_segmentation('0-0000+++0+-++++++000000000+--+-0+'))
	print(find_optimal_segmentation('+0+++---+-'))
	print(find_optimal_segmentation('0+0000---0-0--'))
	
	
	
	
	
	
	
	
	
	
	
	
	
