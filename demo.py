from substitution import Substitution

# change to directory of folder containing needed files
directory = '/Users/hisham/Google Drive/Recipes1M'
sub = Substitution(directory)

recipe = input('Enter your recipe (q to quit): ')
while recipe != 'q':
	print(sub.get_substitutions(recipe.split()))
	recipe = input('Enter recipe ingredients (q to quit): ')