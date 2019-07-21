import sys
import json
import re

JOINT_NAMES = ['spine base', 'spine middle', 'neck', 'head', 'left shoulder', 'left elbow', 'left wrist', 'left hand', 'right shoulder',
				'right elbow', 'right wrist', 'right hand', 'left hip', 'left knee', 'left ankle', 'left foot', 'right hip', 'right knee',
				'right ankle', 'right foot', 'spine shoulder', 'left hand tip', 'left thumb', 'right hand tip', 'right thumb']
LOCATION_PROBLEMS = {('minus', 'x') : 'left', ('minus', 'y') : 'low', 
					('minus', 'z') : 'backwards', ('plus', 'x') : 'right',
					('plus', 'y') : 'up', ('plus', 'z') : 'forwards'}
					
UNSTABLE_PROBLEMS = {'x' : 'horizontal', 'y' : 'vertical', 'z' : 'depth'}

loc_or_grad_xtor = re.compile(r'^(\d+),(location|gradient)_(x|y|z)$')
angle_xtor = re.compile(r'^\[(\d+), (\d+), (\d+)\],joint_angles$')
dist_xtor = re.compile(r'^(\d+),(\d+),joint_distance$')

def generate_feedback_text(obj, fps):
	start_sec = obj['seq']['start'] / float(fps)
	end_sec = obj['seq']['end'] / float(fps)
	
	suffix = 'from {0:.2f}s to {1:.2f}s'.format(start_sec, end_sec)
	res_loc_or_grad = loc_or_grad_xtor.search(obj['key'])
	res_angle = angle_xtor.search(obj['key'])
	res_dist = dist_xtor.search(obj['key'])
	if obj['key'] == 'video_length_frames_raw':
		return 'The motion duration is too {0}'.format('short' if obj['seq']['classification'] == 'minus' else 'long')
	if obj['key'] == 'sequence_length':
		return 'The motion is too {0}, {1}'.format('fast' if obj['seq']['classification'] == 'minus' else 'slow', suffix)
	if res_loc_or_grad != None:
		pattern = 'The {0} {1}, {2}'
		joint = JOINT_NAMES[int(res_loc_or_grad.group(1))]
		loc_or_grad = res_loc_or_grad.group(2)
		axe = res_loc_or_grad.group(3)
		if loc_or_grad == 'location':
			if obj['seq']['classification'] == 'mult':
				problem = 'is unstable in the {0} direction'.format(UNSTABLE_PROBLEMS[axe])
			else:
				problem = 'is too {0}'.format(LOCATION_PROBLEMS[(obj['seq']['classification'], axe)])
		elif loc_or_grad == 'gradient':
			move_type = 'in an unstable velocity' if obj['seq']['classification'] == 'mult' else ('too fast' if obj['seq']['classification'] == 'plus' else 'too slow')
			problem = 'moves {0} in the {1} direction'.format(move_type, UNSTABLE_PROBLEMS[axe])
			
		return pattern.format(joint, problem, suffix)
	if res_dist != None:
		pattern = 'The {0} {1} the {2}, {3}'
		joint_1 = JOINT_NAMES[int(res_dist.group(1))]
		joint_2 = JOINT_NAMES[int(res_dist.group(2))]
		if obj['seq']['classification'] == 'mult':
			problem = 'has unstable distance from'
		elif obj['seq']['classification'] == 'plus':
			problem = 'is too distant from'
		elif obj['seq']['classification'] == 'minus':
			problem = 'is too close to'
			
		return pattern.format(joint_1, problem, joint_2, suffix)
	if res_angle != None:
		pattern = 'The angle between the {0} and the {1}, through the {2}, is {3}, {4}'
		joint_1 = JOINT_NAMES[int(res_angle.group(1))]
		joint_2 = JOINT_NAMES[int(res_angle.group(2))]
		joint_3 = JOINT_NAMES[int(res_angle.group(3))]
		problem = 'unstable' if obj['seq']['classification'] == 'mult' else ('too {0}'.format('acute' if obj['seq']['classification'] == 'minus' else 'obtuse'))
		
		return pattern.format(joint_1, joint_3, joint_2, problem, suffix)
	raise Exception('Unsupported feedback type')

if __name__ == '__main__':
	input_json_file = sys.argv[1]
	fps = float(sys.argv[2])
	
	with open(input_json_file, 'r') as f:
		data = json.load(f)
	
	print(generate_feedback_text(data, fps))
	