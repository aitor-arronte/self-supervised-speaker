import os
from pydub import AudioSegment


def generate_txt_file(voxceleb_path):
	file_out = open('./dataset/voxceleb2_train.txt', 'w')
	for root, dirs, files in os.walk(voxceleb_path):
		path = root.split(os.sep)
		for file in files:
			file_out.write(path[-2]+' '+root+'/'+file+'\n')
	file_out.close()


def m4a_to_wav(path):
	for root, dirs, files in os.walk(path):
		for file in files:
			if file =='.DS_Store':
				os.remove(root+'/'+file)
				continue
			elif file.endswith('.wav'):
				continue
			source_file = root+'/'+file
			try:
				track = AudioSegment.from_file(source_file)
				track.export(source_file.replace('.m4a', '.wav'), format='wav')
				os.remove(source_file)
			except:
				print("ERROR WITH FILE" + str(source_file))