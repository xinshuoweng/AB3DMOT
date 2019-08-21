import os, copy, glob, glob2, numpy as np, colorsys

def isstring(string_test):
	try:
		return isinstance(string_test, basestring)
	except NameError:
		return isinstance(string_test, str)

def islist(list_test):
	return isinstance(list_test, list)

def islogical(logical_test):
	return isinstance(logical_test, bool)

def isnparray(nparray_test):
	return isinstance(nparray_test, np.ndarray)

# def istuple(tuple_test):
# 	return isinstance(tuple_test, tuple)

# def isfunction(func_test):
# 	return callable(func_test)

# def isdict(dict_test):
# 	return isinstance(dict_test, dict)

# def isext(ext_test):
# 	'''
# 	check if it is an extension, only '.something' is an extension, multiple extension is not a valid extension
# 	'''
# 	return isstring(ext_test) and ext_test[0] == '.' and len(ext_test) > 1 and ext_test.count('.') == 1

def isinteger(integer_test):
	if isnparray(integer_test): return False
	try: return isinstance(integer_test, int) or int(integer_test) == integer_test
	except (TypeError, ValueError): return False

def is_path_valid(pathname):
	try:
		if not isstring(pathname) or not pathname: return False
	except TypeError: return False
	else: return True

def is_path_creatable(pathname):
	'''
	if any previous level of parent folder exists, returns true
	'''
	if not is_path_valid(pathname): return False
	pathname = os.path.normpath(pathname)
	pathname = os.path.dirname(os.path.abspath(pathname))

	# recursively to find the previous level of parent folder existing
	while not is_path_exists(pathname):
		pathname_new = os.path.dirname(os.path.abspath(pathname))
		if pathname_new == pathname: return False
		pathname = pathname_new
	return os.access(pathname, os.W_OK)

def is_path_exists(pathname):
	try: return is_path_valid(pathname) and os.path.exists(pathname)
	except OSError: return False

def is_path_exists_or_creatable(pathname):
	try: return is_path_exists(pathname) or is_path_creatable(pathname)
	except OSError: return False

def isfolder(pathname):
	'''
	if '.' exists in the subfolder, the function still justifies it as a folder. e.g., /mnt/dome/adhoc_0.5x/abc is a folder
	if '.' exists after all slashes, the function will not justify is as a folder. e.g., /mnt/dome/adhoc_0.5x is NOT a folder
	'''
	if is_path_valid(pathname):
		pathname = os.path.normpath(pathname)
		if pathname == './': return True
		name = os.path.splitext(os.path.basename(pathname))[0]
		ext = os.path.splitext(pathname)[1]
		return len(name) > 0 and len(ext) == 0
	else: return False

def safe_path(input_path, warning=True, debug=True):
    '''
    convert path to a valid OS format, e.g., empty string '' to '.', remove redundant '/' at the end from 'aa/' to 'aa'

    parameters:
    	input_path:		a string

    outputs:
    	safe_data:		a valid path in OS format
    '''
    if debug: assert isstring(input_path), 'path is not a string: %s' % input_path
    safe_data = copy.copy(input_path)
    safe_data = os.path.normpath(safe_data)
    return safe_data

def fileparts(input_path, warning=True, debug=True):
	'''
	this function return a tuple, which contains (directory, filename, extension)
	if the file has multiple extension, only last one will be displayed

	parameters:
		input_path:     a string path

	outputs:
		directory:      the parent directory
		filename:       the file name without extension
		ext:            the extension
	'''
	good_path = safe_path(input_path, debug=debug)
	if len(good_path) == 0: return ('', '', '')
	if good_path[-1] == '/':
		if len(good_path) > 1: return (good_path[:-1], '', '')	# ignore the final '/'
		else: return (good_path, '', '')	                          # ignore the final '/'

	directory = os.path.dirname(os.path.abspath(good_path))
	filename = os.path.splitext(os.path.basename(good_path))[0]
	ext = os.path.splitext(good_path)[1]
	return (directory, filename, ext)

def mkdir_if_missing(input_path, warning=True, debug=True):
	'''
	create a directory if not existing:
		1. if the input is a path of file, then create the parent directory of this file
		2. if the root directory does not exists for the input, then create all the root directories recursively until the parent directory of input exists

	parameters:
		input_path:     a string path
	'''
	good_path = safe_path(input_path, warning=warning, debug=debug)
	if debug: assert is_path_exists_or_creatable(good_path), 'input path is not valid or creatable: %s' % good_path
	dirname, _, _ = fileparts(good_path)
	if not is_path_exists(dirname): mkdir_if_missing(dirname)
	if isfolder(good_path) and not is_path_exists(good_path): os.mkdir(good_path)

def load_txt_file(file_path, debug=True):
    '''
    load data or string from text file
    '''
    file_path = safe_path(file_path)
    if debug: assert is_path_exists(file_path), 'text file is not existing at path: %s!' % file_path
    with open(file_path, 'r') as file: data = file.read().splitlines()
    num_lines = len(data)
    file.close()
    return data, num_lines

def load_list_from_folder(folder_path, ext_filter=None, depth=1, recursive=False, sort=True, save_path=None, debug=True):
    '''
    load a list of files or folders from a system path

    parameters:
        folder_path:    root to search
        ext_filter:     a string to represent the extension of files interested
        depth:          maximum depth of folder to search, when it's None, all levels of folders will be searched
        recursive:      False: only return current level
                        True: return all levels till to the input depth

    outputs:
        fulllist:       a list of elements
        num_elem:       number of the elements
    '''
    folder_path = safe_path(folder_path)
    if debug: assert isfolder(folder_path), 'input folder path is not correct: %s' % folder_path
    if not is_path_exists(folder_path):
        print('the input folder does not exist\n')
        return [], 0
    if debug:
        assert islogical(recursive), 'recursive should be a logical variable: {}'.format(recursive)
        assert depth is None or (isinteger(depth) and depth >= 1), 'input depth is not correct {}'.format(depth)
        assert ext_filter is None or (islist(ext_filter) and all(isstring(ext_tmp) for ext_tmp in ext_filter)) or isstring(ext_filter), 'extension filter is not correct'
    if isstring(ext_filter): ext_filter = [ext_filter]                               # convert to a list
    # zxc

    fulllist = list()
    if depth is None:        # find all files recursively
        recursive = True
        wildcard_prefix = '**'
        if ext_filter is not None:
            for ext_tmp in ext_filter:
                # wildcard = os.path.join(wildcard_prefix, '*' + string2ext_filter(ext_tmp))
                wildcard = os.path.join(wildcard_prefix, '*' + ext_tmp)
                curlist = glob2.glob(os.path.join(folder_path, wildcard))
                if sort: curlist = sorted(curlist)
                fulllist += curlist
        else:
            wildcard = wildcard_prefix
            curlist = glob2.glob(os.path.join(folder_path, wildcard))
            if sort: curlist = sorted(curlist)
            fulllist += curlist
    else:                    # find files based on depth and recursive flag
        wildcard_prefix = '*'
        for index in range(depth-1): wildcard_prefix = os.path.join(wildcard_prefix, '*')
        if ext_filter is not None:
            for ext_tmp in ext_filter:
                # wildcard = wildcard_prefix + string2ext_filter(ext_tmp)
                wildcard = wildcard_prefix + ext_tmp
                curlist = glob.glob(os.path.join(folder_path, wildcard))
                if sort: curlist = sorted(curlist)
                fulllist += curlist
            # zxc
        else:
            wildcard = wildcard_prefix
            curlist = glob.glob(os.path.join(folder_path, wildcard))
            # print(curlist)
            if sort: curlist = sorted(curlist)
            fulllist += curlist
        if recursive and depth > 1:
            newlist, _ = load_list_from_folder(folder_path=folder_path, ext_filter=ext_filter, depth=depth-1, recursive=True)
            fulllist += newlist

    fulllist = [os.path.normpath(path_tmp) for path_tmp in fulllist]
    num_elem = len(fulllist)

    # save list to a path
    if save_path is not None:
        save_path = safe_path(save_path)
        if debug: assert is_path_exists_or_creatable(save_path), 'the file cannot be created'
        with open(save_path, 'w') as file:
            for item in fulllist: file.write('%s\n' % item)
        file.close()

    return fulllist, num_elem

###################################################### visualization
def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / float(N), 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    # random.shuffle(colors)
    return colors