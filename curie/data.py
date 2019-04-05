# Module for handling data

import csv

def read(fname):
	content = []
	with open(fname, "rb") as f:
		reader = csv.reader(f, delimiter=' ', quotechar='|')
		for row in reader:
			content.append(row)
	return content

def save(fname, sdata):
	with open(fname, 'wb') as f:
		writer = csv.writer(f, delimiter=' ',quotechar='|', \
		quoting=csv.QUOTE_MINIMAL)
		for row in sdata:
			writer.writerow(row)