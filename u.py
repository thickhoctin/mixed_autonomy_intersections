import subprocess, sys, os, re, tempfile, zipfile, gzip, io, shutil, string, random, itertools, pickle, json, yaml, gc, inspect
from itertools import chain, groupby, islice, product, permutations, combinations
from datetime import datetime
from time import time
from fnmatch import fnmatch
from glob import glob
from tqdm import tqdm
from copy import copy, deepcopy
from collections import OrderedDict, defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

version = sys.version_info
if version[0] < 3:
    from StringIO import StringIO
else:
    from io import StringIO

def lrange(*args, **kwargs):
    return list(range(*args, **kwargs))

def lchain(*args):
    return list(chain(*args))

def lmap(fn, *iterables):
    return [fn(*xs) for xs in zip(*iterables)]

def lif(keep, *x):
    return x if keep else []

def dif(keep, **kwargs):
    return kwargs if keep else {}

def flatten(x):
    return [z for y in x for z in y]

def groupby_(xs, key=None):
    if callable(key):
        key = map(key, xs)
    elif key is None:
        key = xs
    groups = defaultdict(list)
    for k, v in zip(key, xs):
        groups[k].append(v)
    return groups

class Dict(dict if version.major == 3 and version.minor >= 6 else OrderedDict):
    def __add__(self, d):
        return Dict(**self).merge(d)

    def merge(self, *dicts, **kwargs):
        for d in dicts:
            self.update(d)
        self.update(kwargs)
        return self

    def filter(self, keys):
        try: # check for iterable
            keys = set(keys)
            return Dict((k, v) for k, v in self.items() if k in keys)
        except TypeError: # function key
            f = keys
            return Dict((k, v) for k, v in self.items() if f(k, v))

    def map(self, mapper):
        if callable(mapper): # function mapper
            return Dict((k, mapper(v)) for k, v in self.items())
        else: # dictionary mapper
            return Dict((k, mapper[v]) for k, v in self.items())

def load_json(path):
    with open(path, 'r+') as f:
        return json.load(f)

def save_json(path, dict_):
    with open(path, 'w+') as f:
        json.dump(dict_, f, indent=4, sort_keys=True)

def format_json(dict_):
    return json.dumps(dict_, indent=4, sort_keys=True)

def format_yaml(dict_):
    dict_ = recurse(dict_, lambda x: x._ if isinstance(x, Path) else dict(x) if isinstance(x, Dict) else x)
    return yaml.dump(dict_)

def load_text(path, encoding='utf-8'):
    with open(path, 'r', encoding=encoding) as f:
        return f.read()

def save_text(path, string):
    with open(path, 'w') as f:
        f.write(string)

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_pickle(path, obj):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def wget(link, output_dir):
    cmd = 'wget %s -P %s' % (link, output_dir)
    shell(cmd)
    output_path = Path(output_dir) / os.path.basename(link)
    if not output_path.exists(): raise RuntimeError('Failed to run %s' % cmd)
    return output_path

def extract(input_path, output_path=None):
    if input_path[-3:] == '.gz':
        if not output_path:
            output_path = input_path[:-3]
        with gzip.open(input_path, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                f_out.write(f_in.read())
    else:
        raise RuntimeError('Don\'t know file extension for ' + input_path)

def rand_string(length):
    import string
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))

nexti = nextk = lambda iterable: next(iter(iterable))
nextv = lambda dict: next(iter(dict.values()))
nextkv = lambda dict: next(iter(dict.items()))

def shell(cmd, wait=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE):
    stdout = stdout or subprocess.DEVNULL
    stderr = stderr or subprocess.DEVNULL
    if not isinstance(cmd, str):
        cmd = ' '.join(cmd)
    process = subprocess.Popen(cmd, shell=True, stdout=stdout, stderr=stderr)
    if not wait:
        return process
    out, err = process.communicate()
    return out.decode().rstrip('\n') if out else '', err.decode().rstrip('\n') if err else ''

def terminal_height():
    return int(shell('tput lines')[0])

def terminal_width():
    return int(shell('tput cols')[0])

def git_state(dir=None):
    cwd = os.getcwd()
    dir = dir or shell('git rev-parse --show-toplevel')[0]
    os.chdir(dir)
    status = shell('git status')[0]
    base_commit = shell('git rev-parse HEAD')[0]
    diff = shell('git diff %s' % base_commit)[0]
    os.chdir(cwd)
    return base_commit, diff, status

def attrs(obj):
    for k, v in inspect.getmembers(obj):
        if inspect.isfunction(v) or inspect.ismethod(v):
            print(f'{v.__name__}{inspect.signature(v)}')
        elif not callable(v) and not k.startswith('__'):
            print(k, v)

def source(obj):
    print(inspect.getsource(obj))

def import_module(module_name, module_path):
    import imp
    module = imp.load_source(module_name, module_path)
    return module

def str2num(s):
    try: return int(s)
    except:
        try: return float(s)
        except: return s

def get_time_log_path():
    return datetime.now().isoformat().replace(':', '_').rsplit('.')[0] + '.log'

_log_path = None
def logger(directory=None):
    global _log_path
    if directory and not _log_path:
        from datetime import datetime
        _log_path = Path(directory) / get_time_log_path()
    return log

def log(text):
    print(text)
    if _log_path:
        with open(_log_path, 'a') as f:
            f.write(text)
            f.write('\n')

def installed(pkg):
    out, err = shell('dpkg -l %s' % pkg)
    if err and err.startswith('dpkg-query: no packages found matching'):
        return False
    return True

def install(pkgs, root):
    root = Path(root)
    old_cwd = os.getcwd()
    self_installed = set()
    os.chdir(root)
    while pkgs:
        pkg = pkgs.pop()
        print('Processing %s' % pkg)
        if installed(pkg) or pkg in self_installed:
            continue
        out, err = shell('apt-cache depends %s' % pkg)
        deps = []
        for x in out.split('\n'):
            x = x.lstrip()
            if x.startswith('Depends:'):
                splits = x.split(' ')
                assert len(splits) == 2
                dep = splits[1]
                if not (dep in self_installed or installed(dep)):
                    deps.append(dep)
        print('Found needed dependencies %s for %s' % (deps, pkg))
        pkgs.extend(deps)
        tmp = Path('tmp')
        shell('mkdir tmp && cd tmp && apt download %s' % pkg)
        for deb in tmp.glob('*.deb'):
            shell('dpkg -x %s .' % deb)
            print('Installing %s with %s' % (pkg, deb))
            self_installed.add(pkg)
        tmp.rm()
    lib = Path('usr/lib')
    real_root = Path('/')
    for x in lib, lib / 'x86_64-linux-gnu':
        brokens = x.lslinks(exist=False)
        for broken in brokens:
            real = real_root / broken._up / os.readlink(broken)
            if real.exists():
                broken.link(real, force=True)
                print('Fixing broken link to be %s -> %s' % (broken, real))
            else:
                print('Could not fix broken link %s' % broken)
    os.chdir(old_cwd)

class Path(str):
    """"""
    @classmethod
    def env(cls, var):
        return Path(os.environ[var])

    def __init__(self, path):
        pass

    def __add__(self, subpath):
        return Path(str(self) + str(subpath))

    def __truediv__(self, subpath):
        return Path(os.path.join(str(self), str(subpath)))

    def __floordiv__(self, subpath):
        return (self / subpath)._

    def ls(self, show_hidden=True, dir_only=False, file_only=False):
        subpaths = [Path(self / subpath) for subpath in os.listdir(self) if show_hidden or not subpath.startswith('.')]
        isdirs = [os.path.isdir(subpath) for subpath in subpaths]
        subdirs = [subpath for subpath, isdir in zip(subpaths, isdirs) if isdir]
        files = [subpath for subpath, isdir in zip(subpaths, isdirs) if not isdir]
        if dir_only:
            return subdirs
        if file_only:
            return files
        return subdirs, files

    def lsdirs(self, show_hidden=True):
        return self.ls(show_hidden=show_hidden, dir_only=True)

    def lsfiles(self, show_hidden=True):
        return self.ls(show_hidden=show_hidden, file_only=True)

    def lslinks(self, show_hidden=True, exist=None):
        '''
        Description: Lists symbolic links in the directory, optionally filtering by existence.
        @param show_hidden: Whether to include hidden files/dirs.
        @param exist: If set, filter by whether the link exists.
        @return: List of Path objects that are symlinks.
        '''
        dirs, files = self.ls(show_hidden=show_hidden)
        return [x for x in dirs + files if x.islink() and (
            exist is None or not (exist ^ x.exists()))]

    def glob(self, glob_str):
        '''
        Description: Returns a list of Paths matching a glob pattern.
        @param glob_str: Glob pattern string.
        @return: List of Path objects.
        '''
        return [Path(p) for p in glob(self / glob_str, recursive=True)]

    def re(self, re_pattern):
        '''
        Description: Returns a list of Paths matching a regex pattern (directory tree).
        @param re_pattern: Regex pattern string (split by '/').
        @return: List of Path objects.
        '''
        subpatterns = lmap(re.compile, re_pattern.split('/'))
        matches = []
        dirs, files = self.ls()
        for pattern in subpatterns[:-1]:
            new_dirs, new_files = [], []
            for d in filter(lambda x: pattern.fullmatch(x._name), dirs):
                d_dirs, d_files = d.ls()
                new_dirs.extend(d_dirs)
                new_files.extend(d_files)
            dirs, files = new_dirs, new_files
        return sorted(filter(lambda x: subpatterns[-1].fullmatch(x._name), dirs + files))

    def recurse(self, dir_fn=None, file_fn=None):
        '''
        Description: Recursively applies functions to all subdirs and files in directory.
        @param dir_fn: Function to apply to directories.
        @param file_fn: Function to apply to files.
        @return: None
        '''
        if dir_fn is not None:
            dir_fn(self)
        dirs, files = self.ls()
        if file_fn is not None:
            list(map(file_fn, files))
        for dir in dirs:
            dir.recurse(dir_fn=dir_fn, file_fn=file_fn)

    def mk(self):
        '''
        Description: Creates the directory (including parents).
        @return: Self (Path object).
        '''
        os.makedirs(self, exist_ok=True)
        return self

    def dir_mk(self):
        '''
        Description: Creates the parent directory of this path.
        @return: Self (Path object).
        '''
        self._up.mk()
        return self

    def rm(self):
        '''
        Description: Removes the file or directory at this path.
        @return: Self (Path object).
        '''
        if self.isfile() or self.islink():
            os.remove(self)
        elif self.isdir():
            shutil.rmtree(self)
        return self

    def unlink(self):
        '''
        Description: Removes a symbolic link.
        @return: Self (Path object).
        '''
        os.unlink(self)
        return self


    def mv(self, dest):
        '''
        Description: Moves this file or directory to a new destination.
        @param dest: Destination path.
        @return: None
        '''
        shutil.move(self, dest)

    def mv_from(self, src):
        '''
        Description: Moves a file or directory from src to this path.
        @param src: Source path.
        @return: None
        '''
        shutil.move(src, self)

    def cp(self, dest):
        '''
        Description: Copies this file to a new destination.
        @param dest: Destination path.
        @return: None
        '''
        shutil.copy(self, dest)

    def cp_from(self, src):
        '''
        Description: Copies a file from src to this path.
        @param src: Source path.
        @return: None
        '''
        shutil.copy(src, self)

    def link(self, target, force=False):
        '''
        Description: Creates a symbolic link at this path to the target.
        @param target: Target path.
        @param force: If True, overwrite existing link.
        @return: None
        '''
        if self.lexists():
            if not force:
                return
            else:
                self.rm()
        os.symlink(target, self)

    def exists(self):
        '''
        Description: Checks if the path exists.
        @return: True if exists, else False.
        '''
        return os.path.exists(self)

    def lexists(self):
        '''
        Description: Checks if the path exists (including broken symlinks).
        @return: True if exists, else False.
        '''
        return os.path.lexists(self)

    def isfile(self):
        '''
        Description: Checks if the path is a file.
        @return: True if file, else False.
        '''
        return os.path.isfile(self)

    def isdir(self):
        '''
        Description: Checks if the path is a directory.
        @return: True if directory, else False.
        '''
        return os.path.isdir(self)

    def islink(self):
        '''
        Description: Checks if the path is a symbolic link.
        @return: True if symlink, else False.
        '''
        return os.path.islink(self)

    def chdir(self):
        '''
        Description: Changes the current working directory to this path.
        @return: None
        '''
        os.chdir(self)

    def rel(self, start=None):
        '''
        Description: Returns the relative path from start to this path.
        @param start: Start path (optional).
        @return: Path object (relative path).
        '''
        return Path(os.path.relpath(self, start=start))

    def clone(self):
        '''
        Description: Returns a new Path with an incremented suffix if the path exists.
        @return: New Path object.
        '''
        name = self._name
        match = re.search('__([0-9]+)$', name)
        if match is None:
            base = self + '__'
            i = 1
        else:
            initial = match.group(1)
            base = self[:-len(initial)]
            i = int(initial) + 1
        while True:
            path = Path(base + str(i))
            if not path.exists():
                return path
            i += 1

    @property
    def _(self):
        '''
        Description: Returns the string representation of the path.
        @return: String path.
        '''
        return str(self)

    @property
    def _real(self):
        '''
        Description: Returns the absolute, expanded path.
        @return: Path object (absolute path).
        '''
        return Path(os.path.realpath(os.path.expanduser(self)))

    @property
    def _up(self):
        '''
        Description: Returns the parent directory as a Path.
        @return: Path object (parent directory).
        '''
        path = os.path.dirname(self.rstrip('/'))
        if path == '':
            path = os.path.dirname(self._real.rstrip('/'))
        return Path(path)

    @property
    def _name(self):
        '''
        Description: Returns the base name of the path.
        @return: Path object (base name).
        '''
        return Path(os.path.basename(self))

    @property
    def _stem(self):
        '''
        Description: Returns the stem (filename without extension).
        @return: Path object (stem).
        '''
        return Path(os.path.splitext(self)[0])

    @property
    def _basestem(self):
        '''
        Description: Returns the base stem (recursively removes extensions).
        @return: Path object (base stem).
        '''
        new = self._stem
        while new != self:
            new, self = new._stem, new
        return new

    @property
    def _ext(self):
        '''
        Description: Returns the file extension.
        @return: Path object (extension).
        '''
        return Path(os.path.splitext(self)[1])

    extract = extract
    load_json = load_json
    save_json = save_json
    load_txt = load_sh = load_text
    save_txt = save_sh = save_text
    load_p = load_pickle
    save_p = save_pickle

    def save_bytes(self, bytes):
        '''
        Description: Saves bytes to this file.
        @param bytes: Bytes to write.
        @return: None
        '''
        with open(self, 'wb') as f:
            f.write(bytes)

    def load_csv(self, index_col=0, **kwargs):
        '''
        Description: Loads a CSV file into a pandas DataFrame.
        @param index_col: Index column for DataFrame.
        @param **kwargs: Additional pandas read_csv arguments.
        @return: pandas DataFrame.
        '''
        return pd.read_csv(self, index_col=index_col, **kwargs)

    def save_csv(self, df, float_format='%.5g', **kwargs):
        '''
        Description: Saves a pandas DataFrame to a CSV file.
        @param df: DataFrame to save.
        @param float_format: Float format string.
        @param **kwargs: Additional pandas to_csv arguments.
        @return: None
        '''
        df.to_csv(self, float_format=float_format, **kwargs)

    def load_npy(self):
        '''
        Description: Loads a numpy array from a .npy file.
        @return: numpy array.
        '''
        return np.load(self, allow_pickle=True)

    def save_npy(self, obj):
        '''
        Description: Saves a numpy array to a .npy file.
        @param obj: numpy array to save.
        @return: None
        '''
        np.save(self, obj)

    def load_yaml(self):
        '''
        Description: Loads a YAML file and returns the parsed object.
        @return: Parsed YAML object.
        '''
        with open(self, 'r') as f:
            return yaml.safe_load(f)

    def save_yaml(self, obj):
        '''
        Description: Saves an object as YAML to this file.
        @param obj: Object to save.
        @return: None
        '''
        obj = recurse(obj, lambda x: x._ if isinstance(x, Path) else dict(x) if isinstance(x, Dict) else x)
        with open(self, 'w') as f:
            yaml.dump(obj, f, default_flow_style=False, allow_unicode=True)

    def load_pth(self):
        '''
        Description: Loads a PyTorch model from a .pth file.
        @return: Loaded PyTorch object.
        '''
        return torch.load(self)

    def save_pth(self, obj):
        '''
        Description: Saves a PyTorch object to a .pth file.
        @param obj: PyTorch object to save.
        @return: None
        '''
        torch.save(obj, self)

    def load_pdf(self):
        '''
        Description: Loads a PDF file and returns a PdfReader object.
        @return: PdfReader object.
        Can use index and slice obj.pages for the pages, then call Path.save_pdf to save
        '''
        from pdfrw import PdfReader
        return PdfReader(self)

    def save_pdf(self, pages):
        '''
        Description: Saves PDF pages to this file.
        @param pages: PDF pages to save.
        @return: None
        '''
        from pdfrw import PdfWriter
        writer = PdfWriter()
        writer.addpages(pages)
        writer.write(self)

    def load(self):
        '''
        Description: Loads an object from this file, inferring the format from the extension.
        @return: Loaded object.
        '''
        return eval('self.load_%s' % self._ext[1:])()

    def save(self, obj):
        '''
        Description: Saves an object to this file, inferring the format from the extension.
        @param obj: Object to save.
        @return: None
        '''
        return eval('self.save_%s' % self._ext[1:])(obj)

    def replace_txt(self, replacements, dst=None):
        '''
        Description: Replaces text in the file according to a replacements dict and saves.
        @param replacements: Dict of {old: new} replacements.
        @param dst: Destination path (optional).
        @return: None
        '''
        content = self.load_txt()
        for k, v in replacements.items():
            content = content.replace(k, v)
        (dst or self).save_txt(content)

    def update_dict(self, updates={}, vars=[], unvars=[], dst=None):
        '''
        Description: Updates a dict stored in this file, optionally setting/unsetting keys, and saves.
        @param updates: Dict of updates.
        @param vars: List of keys to set True.
        @param unvars: List of keys to remove.
        @param dst: Destination path (optional).
        @return: None
        '''
        d = self.load()
        for k in vars:
            d[k] = True
        for k in unvars:
            d.pop(k, None)
        d.update(updates)
        (dst or self).save(d)

    def torch_strip(self, dst):
        '''
        Description: Removes 'opt' and 'step' keys from a torch checkpoint dict and saves.
        @param dst: Destination path.
        @return: None
        '''
        self.update_dict(unvars=['opt', 'step'], dst=dst)

    def wget(self, link):
        '''
        Description: Downloads a file to this directory using wget.
        @param link: URL to download.
        @return: Path to downloaded file.
        '''
        if self.isdir():
            return Path(wget(link, self))
        raise ValueError('Path %s needs to be a directory' % self)

    def replace(self, old, new=''):
        '''
        Description: Returns a new Path with replaced substring.
        @param old: Substring to replace.
        @param new: Replacement string.
        @return: New Path object.
        '''
        return Path(super().replace(old, new))

    def search(self, pattern):
        '''
        Description: Searches for a regex pattern in the path string.
        @param pattern: Regex pattern.
        @return: Match object or None.
        '''
        return re.search(pattern, self)

    def search_pattern(self, pattern):
        '''
        Description: Returns the first matching group for a regex pattern in the path string.
        @param pattern: Regex pattern.
        @return: String of the first match.
        '''
        return self.search(pattern).group()

    def search_groups(self, pattern):
        '''
        Description: Returns all matching groups for a regex pattern in the path string.
        @param pattern: Regex pattern.
        @return: Tuple of matching groups.
        '''
        return self.search(pattern).groups()

    def search_group(self, pattern):
        '''
        Description: Returns the first group for a regex pattern in the path string.
        @param pattern: Regex pattern.
        @return: String of the first group.
        '''
        return self.search_groups(pattern)[0]

    def findall(self, pattern):
        '''
        Description: Finds all non-overlapping matches of a regex pattern in the path string.
        @param pattern: Regex pattern.
        @return: List of matches.
        '''
        return re.findall(pattern, self)

class Namespace(Dict):
    def __init__(self, *args, **kwargs):
        '''
        Description: Initializes the Namespace with given variables.
        @param *args: Positional arguments (str or dict).
        @param **kwargs: Keyword arguments.
        @return: None
        '''
        self.var(*args, **kwargs)

    def var(self, *args, **kwargs):
        '''
        Description: Sets variables in the Namespace from args and kwargs.
        @param *args: Strings (set True) or dicts (update).
        @param **kwargs: Key-value pairs to set.
        @return: Self (Namespace).
        '''
        kvs = Dict()
        for a in args:
            if isinstance(a, str):
                kvs[a] = True
            else: # a is a dictionary
                kvs.update(a)
        kvs.update(kwargs)
        self.update(kvs)
        return self

    def unvar(self, *args):
        '''
        Description: Removes variables from the Namespace.
        @param *args: Keys to remove.
        @return: Self (Namespace).
        '''
        for a in args:
            self.pop(a)
        return self

    def setdefaults(self, *args, **kwargs):
        '''
        Description: Sets default values for keys not already in the Namespace.
        @param *args: Keys to set True if not present.
        @param **kwargs: Key-value pairs to set if not present.
        @return: Self (Namespace).
        '''
        args = [a for a in args if a not in self]
        kwargs = {k: v for k, v in kwargs.items() if k not in self}
        return self.var(*args, **kwargs)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as e:
            self.__getattribute__(key)

    def __setattr__(self, key, value):
        self[key] = value

##### Functions for compute

using_ipython = True
try:
    _ = get_ipython().__class__.__name__
except NameError:
    using_ipython = False

import numpy as np

import pandas as pd
def _sel(self, col, value):
    if isinstance(value, list):
        return self[self[col].isin(value)]
    return self[self[col] == value]
pd.DataFrame.sel = _sel

import scipy.stats
import scipy as sp
from scipy.stats import pearsonr as pearson, spearmanr as spearman, kendalltau

if not using_ipython:
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

arrayf = lambda *args, **kwargs: np.array(*args, **kwargs, dtype=np.float32)
arrayl = lambda *args, **kwargs: np.array(*args, **kwargs, dtype=np.long)
arrayb = lambda *args, **kwargs: np.array(*args, **kwargs, dtype=np.bool)
arrayo = lambda *args, **kwargs: np.array(*args, **kwargs, dtype=object)

def split(x, sizes):
    '''
    Description: Splits an array into sections with given sizes.
    @param x: Array to split.
    @param sizes: List of sizes for each split.
    @return: List of arrays.
    '''
    return np.split(x, np.cumsum(sizes[:-1]))

def recurse(x, fn):
    '''
    Description: Recursively applies a function to all elements in a nested structure.
    @param x: Input (dict, list, tuple, or value).
    @param fn: Function to apply.
    @return: Structure with function applied to all leaves.
    '''
    if isinstance(x, dict):
        return type(x)((k, recurse(v, fn)) for k, v in x.items())
    elif isinstance(x, (list, tuple)):
        return type(x)(recurse(v, fn) for v in x)
    return fn(x)

def from_numpy(x):
    '''
    Description: Converts numpy arrays to native Python types recursively.
    @param x: Numpy array or nested structure.
    @return: Structure with numpy arrays converted to lists/scalars.
    '''
    def helper(x):
        if type(x).__module__ == np.__name__:
            if isinstance(x, np.ndarray):
                return recurse(list(x), helper)
            return np.asscalar(x)
        return x
    return recurse(x, helper)

def smooth(y, box_pts):
    '''
    Description: Smooths a 1D array using a moving average.
    @param y: Input array.
    @param box_pts: Window size.
    @return: Smoothed array.
    '''
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def gsmooth(y, sigma):
    '''
    Description: Smooths a 1D array using a Gaussian filter.
    @param y: Input array.
    @param sigma: Standard deviation for Gaussian kernel.
    @return: Smoothed array.
    '''
    from scipy.ndimage.filters import gaussian_filter1d
    return gaussian_filter1d(y, sigma=sigma)

def normalize(x, eps=1e-8):
    '''
    Description: Normalizes an array to zero mean and unit variance.
    @param x: Input array.
    @param eps: Small value to avoid division by zero.
    @return: Normalized array.
    '''
    return (x - x.mean()) / x.std()

def inverse_map(arr):
    '''
    Description: Returns the inverse mapping of an array of indices.
    @param arr: Array of indices.
    @return: Inverse mapping array.
    '''
    inv_map = np.zeros(len(arr))
    inv_map[arr] = np.arange(len(arr))
    return inv_map

def pad_arrays(arrs, value):
    '''
    Description: Pads arrays to the same length with a given value.
    @param arrs: List of arrays.
    @param value: Padding value.
    @return: 2D numpy array.
    '''
    max_len = max(len(x) for x in arrs)
    return np.array([np.concatenate([x, np.full(max_len - len(x), value)]) for x in arrs])

def sorted_segment_maps(segments):
    '''
    Description: Computes sorted segment indices and related mappings.
    @param segments: Array of segment lengths.
    @return: Namespace with segment mappings and indices.
    '''
    r = Namespace()
    r.segment_idxs = np.argsort(segments)
    starts = np.cumsum(segments) - segments
    starts = starts[r.segment_idxs]
    r.segments = segments[r.segment_idxs]

    r.unit_idxs = np.array([i for s, length in zip(starts, r.segments) for i in range(s, s + length)])
    r.unit_idxs_r = inverse_map(r.unit_idxs)
    r.segment_uniques, r.segment_blocks, r.segment_counts = zip(*((seg, sum(segs), len(list(segs))) for seg, segs in groupby(r.segments)))
    return r

def get_gpu_info(ssh_fn=lambda x: x):
    '''
    Description: Gets GPU memory and utilization info using nvidia-smi.
    @param ssh_fn: Function to run shell command (default: identity).
    @return: DataFrame with GPU info.
    '''
    nvidia_str, _ = shell(ssh_fn('nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,nounits'))
    nvidia_str = nvidia_str.replace('[Not Supported]', '100').replace(', ', ',')
    nvidia_str_io = StringIO(nvidia_str)

    gpu_df = pd.read_csv(nvidia_str_io, index_col=0)
    devices_str = os.environ.get('CUDA_VISIBLE_DEVICES')
    if devices_str:
        devices = list(map(int, devices_str.split(',')))
        gpu_df = gpu_df.loc[devices]
        gpu_df.index = gpu_df.index.map({k: i for i, k in enumerate(devices)})

    out_df = pd.DataFrame(index=gpu_df.index)
    out_df['memory_total'] = gpu_df['memory.total [MiB]']
    out_df['memory_used'] = gpu_df['memory.used [MiB]']
    out_df['memory_free'] = out_df['memory_total'] - out_df['memory_used']
    out_df['utilization'] = gpu_df['utilization.gpu [%]'] / 100
    out_df['utilization_free'] = 1 - out_df['utilization']
    return out_df

def get_process_gpu_info(pid=None, ssh_fn=lambda x: x):
    '''
    Description: Gets GPU info for a specific process using nvidia-smi.
    @param pid: Process ID (default: None for all, -1 for current process).
    @param ssh_fn: Function to run shell command (default: identity).
    @return: DataFrame or Series with GPU info for the process.
    '''
    nvidia_str, _ = shell(ssh_fn('nvidia-smi --query-compute-apps=pid,gpu_name,used_gpu_memory --format=csv,nounits'))
    nvidia_str_io = StringIO(nvidia_str.replace(', ', ','))

    gpu_df = pd.read_csv(nvidia_str_io, index_col=0)
    if pid is None:
        return gpu_df
    if pid == -1:
        pid = os.getpid()
    return gpu_df.loc[pid]

##### torch functions

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def to_torch(x, device='cuda' if torch.cuda.is_available() else 'cpu', **kwargs):
    def helper(x):
        if x is None:
            return None
        elif isinstance(x, torch.Tensor):
            return x.to(device=device, **kwargs)
        elif np.isscalar(x):
            return x
        return torch.from_numpy(x).to(device=device, **kwargs)
    return recurse(x, helper)

def from_torch(t, force_scalar=False):
    def helper(t):
        if not isinstance(t, torch.Tensor):
            return t
        x = t.detach().cpu().numpy()
        if force_scalar and (x.size == 1 or np.isscalar(x)):
            return np.asscalar(x)
        return x
    return recurse(t, helper)

def count_params(network, requires_grad=False):
    return sum(p.numel() for p in network.parameters() if not requires_grad or p.requires_grad)

def report_memory(device=None, max=False):
    if device:
        device = torch.device(device)
        if max:
            alloc = torch.cuda.max_memory_allocated(device=device)
        else:
            alloc = torch.cuda.memory_allocated(device=device)
        alloc /=  1024 ** 2
        print('%.3f MBs' % alloc)
        return alloc

    numels = Counter()
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            print(type(obj), obj.size())
            numels[obj.device] += obj.numel()
    print()
    for device, numel in sorted(numels.items()):
        print('%s: %s elements, %.3f MBs' % (str(device), numel, numel * 4 / 1024 ** 2))

def clear_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            obj.cpu()
    gc.collect()
    torch.cuda.empty_cache()

def main_only(method):
    def wrapper(self, *args, **kwargs):
        if self.main:
            return method(self, *args, **kwargs)
    return wrapper

class Config(Namespace):
    def __init__(self, res, *args, **kwargs):
        self.res = Path(res)._real
        super(Config, self).__init__()
        self.load()
        self.var(*args, **kwargs)
        self.setdefaults(
            name=self.res._real._name,
            main=True,
            logger=True,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            debug=False,
            opt_level='O0',
            disable_amp=False
        )

    def __repr__(self):
        return format_yaml(dict(self))

    def __hash__(self):
        return hash(repr(self))

    @property
    def path(self):
        return self.res / 'config.yaml'

    def load(self):
        if self.path.exists():
            for k, v in self.path.load().items():
                self[k] = v
        return self

    never_save = {'res', 'name', 'main', 'logger', 'distributed', 'parallel', 'device', 'steps', 'debug'}
    @property
    def attrs_save(self):
        return {k: v for k, v in self.items() if k not in self.never_save}

    def save(self, force=False):
        if force or not self.path.exists():
            self.res.mk()
            self.path.save(from_numpy(self.attrs_save))
        return self

    @classmethod
    def from_args(cls, *globals_locals):
        import argparse
        parser = argparse.ArgumentParser(description='Model arguments')
        parser.add_argument('res', type=Path, help='Result directory')
        parser.add_argument('kwargs', nargs='*', help='Extra arguments that goes into the config')

        args = parser.parse_args()

        kwargs = {}
        for kv in args.kwargs:
            splits = kv.split('=')
            if len(splits) == 1:
                v = True
            else:
                v = splits[1]
                try:
                    v = eval(v, *globals_locals)
                except (SyntaxError, NameError):
                    pass
            kwargs[splits[0]] = v

        return cls(args.res, **kwargs).save()

    def try_save_commit(self, base_dir=None):
        base_commit, diff, status = git_state(base_dir)

        save_dir = (self.res / 'commit').mk()
        (save_dir / 'hash.txt').save(base_commit)
        (save_dir / 'diff.txt').save(diff)
        (save_dir / 'status.txt').save(status)
        return self

    @main_only
    def log(self, text):
        logger(self.res if self.logger else None)(text)

    ### Train result saving ###

    @property
    def train_results(self):
        return self.res / 'train_results.csv'

    def load_train_results(self):
        if self.train_results.exists():
            return pd.read_csv(self.train_results, index_col=0)
        return None

    @main_only
    def save_train_results(self, results):
        results.to_csv(self.train_results, float_format='%.6g')

    ### Set stopped early ###

    @property
    def stopped_early(self):
        return self.res / 'stopped_early'

    @main_only
    def set_stopped_early(self):
        self.stopped_early.save_txt('')

    ### Set training state ###

    @property
    def training(self):
        return self.res / 'is_training'

    @main_only
    def set_training(self, is_training):
        if is_training:
            if self.main and self.training.exists():
                self.log('Another training is found, continue (yes/n)?')
                ans = input('> ')
                if ans != 'yes':
                    exit()
            self.training.save_txt('')
        else:
            self.training.rm()

    ### Model loading ###

    def init_model(self, net, opt=None, step='max', train=True):
        if train:
            assert not self.training.exists(), 'Training already exists'
        # configure parallel training
        devices = os.environ.get('CUDA_VISIBLE_DEVICES')
        self.n_gpus = 0 if self.device == 'cpu' else 1 if self.device.startswith('cuda:') else len(get_gpu_info()) if devices is None else len(devices.split(','))
        can_parallel = self.n_gpus > 1
        self.setdefaults(distributed=can_parallel) # use distributeddataparallel
        self.setdefaults(parallel=can_parallel and not self.distributed) # use dataparallel
        self.local_rank = 0
        self.world_size = 1 # number of processes
        if self.distributed:
            self.local_rank = int(os.environ['LOCAL_RANK']) # rank of the current process
            self.world_size = int(os.environ['WORLD_SIZE'])
            assert self.world_size == self.n_gpus
            torch.cuda.set_device(self.local_rank)
            torch.distributed.init_process_group(backend='nccl', init_method='env://')
            self.main = self.local_rank == 0

        net.to(self.device)
        if train and not self.disable_amp:
            # configure mixed precision
            net, opt = amp.initialize(net, opt, opt_level=self.opt_level, loss_scale=self.get('loss_scale'), verbosity=0 if self.opt_level == 'O0' else 1)
        step = self.set_state(net, opt=opt, step=step)

        if self.distributed:
            import apex
            net = apex.parallel.DistributedDataParallel(net)
        elif self.parallel:
            net = nn.DataParallel(net)

        if train:
            net.train()
            return net, opt, step
        else:
            net.eval()
            return net, step

    def load_model(self, step='best', train=False):
        '''
        step can be 'best', 'max', an integer, or None
        '''
        model = import_module('model', str(self.model))
        net = model.get_net(self)
        opt = model.get_opt(self, net) if train else None
        return self.init_model(net, opt=opt, step=step, train=train)

    @property
    def models(self):
        return (self.res / 'models').mk()

    def model_save(self, step):
        return self.models / ('model-%s.pth' % step)

    def model_step(self, path):
        m = re.match('.+/model-(\d+)\.pth', path)
        if m:
            return int(m.groups()[0])

    @property
    def model_best(self):
        return self.models / 'best_model.pth'

    @main_only
    def link_model_best(self, model_save):
        self.model_best.rm().link(Path(model_save).rel(self.models))

    def get_saved_model_steps(self):
        _, save_paths = self.models.ls()
        if len(save_paths) == 0:
            return []
        return sorted([x for x in map(self.model_step, save_paths) if x is not None])

    def set_state(self, net, opt=None, step='max', path=None):
        state = self.load_state(step=step, path=path)
        if state is None:
            return 0
        if self.get('append_module_before_load'):
            state['net'] = OrderedDict(('module.' + k, v) for k, v in state['net'].items())
        net.load_state_dict(state['net'])
        if opt:
            if 'opt' in state:
                opt.load_state_dict(state['opt'])
            else:
                self.log('No state for optimizer to load')
        if 'amp' in state and self.opt_level != 'O0':
            amp.load_state_dict(state['amp'])
        return state.get('step', 0)

    @main_only
    def get_state(self, net, opt, step):
        try:
            net_dict = net.module.state_dict()
        except AttributeError:
            net_dict = net.state_dict()
        state = dict(step=step, net=net_dict, opt=opt.state_dict())
        try:
            state['amp'] = amp.state_dict()
        except:
            pass
        return to_torch(state, device='cpu')

    def load_state(self, step='max', path=None):
        '''
        step: best, max, integer, None if path is specified
        path: None if step is specified
        '''
        if path is None:
            if step == 'best':
                path = self.model_best
            else:
                if step == 'max':
                    steps = self.get_saved_model_steps()
                    if len(steps) == 0:
                        return None
                    step = max(steps)
                path = self.model_save(step)
        save_path = Path(path)
        if save_path.exists():
            return to_torch(torch.load(save_path), device=self.device)
        return None

    @main_only
    def save_state(self, step, state, clean=True, link_best=False):
        save_path = self.model_save(step)
        if save_path.exists():
            return save_path
        torch.save(state, save_path)
        self.log('Saved model %s at step %s' % (save_path, step))
        if clean and self.get('max_save'):
            self.clean_models(keep=self.max_save)
        if link_best:
            self.link_model_best(save_path)
            self.log('Linked %s to new saved model %s' % (self.model_best, save_path))
        return save_path
