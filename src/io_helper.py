import os

def check_path(path, create_if_not_exist=True):
	if not os.path.exists(path) and create_if_not_exist == True:
		os.makedirs(path)
		return True
	elif not os.path.exists(path) and create_if_not_exist == False:
		return False

def list_files_in_path_os(path, filename_prefix="", filename_suffix="", recursive=True):
	while path[-1] == "/":
		path = path[:-1]
	all_files = []
	for (dirpath, dirnames, fname) in os.walk(path):
		all_files.extend([dirpath + "/" + el for el in fname if filename_prefix in el and filename_suffix in el])
		if recursive == False:
			break
	all_files = sorted(all_files)
	return all_files