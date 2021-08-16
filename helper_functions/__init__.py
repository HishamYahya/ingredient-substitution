from .text_cleaning import TextCleaner

"""
Takes in a length of a list and returns a list of index tuples covering k chunks
"""
def split_array_ranges(length, k):
	chunks = []
	step = int(length/k)
	start_ind = 0
	end_ind = step
	while end_ind < length:
		chunks.append((start_ind, end_ind))
		start_ind = end_ind
		end_ind += step
	chunks.append((start_ind, length))
	return chunks

def tokenize(recipes):
	return list(map(str.split, recipes))