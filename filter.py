import sys
from curie import cfilter

if __name__ == '__main__':
	source_file = sys.argv[1]
	dest_file = sys.argv[2]
	
	# source_file is the file that contains the training data mixed with the attack points
	# dest_file is where the cleaned data will be saved
	afilter = cfilter.FilterAttack(source_file, dest_file)
	afilter.run()