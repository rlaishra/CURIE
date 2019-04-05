# Curie: A method for protecting SVM Classifier from Poisoning Attack

For more detail about the method refer to [Curie A method for protecting SVM Classifier from Poisoning Attack](http://rickylaishram.com/research/curie16.html)

## Requirements
The required libraries are in `requirements.txt`.
Install with: `pip install -r requirements.txt`

## Data Format
The files are in CSV format. Each row is one data point -- the last value is the class and the rest are the features.

Two examples datasets are included in `data/`. 

`artificial_20_2000.txt` has 2000 data points, 20 features and 2 classes. `artificial_20_2000_attack_50.txt` is has 50 attack points inserted. `artificial_20_2000_cleaned.txt` is the cleaned file.

## Attack Data
The attack data were genererated with [AdversariaLib](https://pralab.diee.unica.it/en/AdversariaLib)

## Running CURIE
`python3 filter.py <source_file> <dest_file>`

`<source_file>` is the file that contains the attack point.
`<dest_file>` is the file where the filtered data will be saved.