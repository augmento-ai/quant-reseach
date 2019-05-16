import os

def check_path(path, create_if_not_exist=True):
	if not os.path.exists(path) and create_if_not_exist == True:
		os.makedirs(path)
		return True
	elif not os.path.exists(path) and create_if_not_exist == False:
		return False