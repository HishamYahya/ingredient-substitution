from substitution import Substitution

# change to directory of folder containing needed files
directory = '/Users/hisham/Google Drive/Recipes1M'
sub= Substitution(directory)

recipe = input('Enter recipe ingredients seperated by spaces (q to quit): ')

def print_substitution(sub):
	print(f"{sub['from']} can be substituted with {sub['to']} which would reduce total recipe ghg emissions by {sub['percent_reduction']}%")
	response = ''
	while response != 'y' and response != 'n':
		response = input('Does that make sense? (y or n): ')
	if response == 'n':
		with open('errors.txt', 'a') as f:
			f.write(f"{recipe},{sub['from']},{sub['to']}\n")
		print('Noted!')
	else:
		print('Great!')	

while recipe != 'q':
	subs = sub.get_substitutions(recipe.split(), verbose=True)
	if not subs:
		print('No valid substitutions found')
	else:
		print_substitution(subs[0])
		for s in subs[1:]:
			print('Also,')
			print_substitution(s)
		
	recipe = input('Enter recipe ingredients (q to quit): ')