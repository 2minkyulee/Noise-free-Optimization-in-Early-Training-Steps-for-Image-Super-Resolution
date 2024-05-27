import warnings
import importlib
import glob
import sys
import os
import traceback
import natsort

LOWER_IS_BETTER = ["LPIPS", "NIQE"]
HIGHER_IS_BETTER = ["PSNR", "SSIM"]

def metric_init(metric):
    
    if metric in LOWER_IS_BETTER:
        return 9999
    
    if metric in HIGHER_IS_BETTER:
        return 0

def is_better(this_is, better_than, metric):
    """
    :param this_is:       dict. ex) {"LPIPS": xx.xx, "PSNR": xx.xx, "SSIM": xx.xx}
    :param better_than:   dict. ex) {"LPIPS": xx.xx, "PSNR": xx.xx, "SSIM": xx.xx}
    :param metric:        on which metric to compare
    :return:
    """
    assert metric in (LOWER_IS_BETTER + HIGHER_IS_BETTER)
    
    if metric in LOWER_IS_BETTER:
        return this_is[metric] < better_than[metric]
    
    else:
        return this_is[metric] > better_than[metric]
        

# Disable
def _blockPrint():
    sys.stdout = open(os.devnull, 'w')
# Restore
def _enablePrint():
    sys.stdout.close()
    sys.stdout = sys.__stdout__

class blockprint(object):
    def __init__(self, activate=True):
        self.activate = activate
        self._blockPrint = _blockPrint
        self._enablePrint = _enablePrint
    def __enter__(self):
        self._blockPrint()
    def __exit__(self, exc_type, exc_value, tb):
        self._enablePrint()
        if exc_type is not None:
            traceback.print_exception(exc_type, exc_value, tb)
            # return False # uncomment to pass exception through
    
        return True
        
class blockwarning(object):
    def __init__(self, activate=True):
        self.activate = True
    def __enter__(self):
        if self.activate:
            warnings.filterwarnings(action="ignore")
    def __exit__(self, exc_type, exc_value, tb):
        pass
        

def get_n_params(model):
    return sum(p.numel() for p in model.parameters())

def get_image_names(path):
    """
    :param path: (list of paths of images, or a path of images) for valid paths, (None, or empty string ("")) for empty paths.
    :return:
    """
    extensions = ["jpg", "png"]
    if isinstance(path, list) or isinstance(path, tuple):
        image_names = []
        for _path in path:
            for extension in extensions:
                
                tmp = natsort.natsorted(glob.glob(os.path.join(_path, f"*.{extension}")))
                image_names += tmp



    elif isinstance(path, str):
        for extension in extensions:
            image_names = natsort.natsorted(glob.glob(os.path.join(path, f"*.{extension}")))

    elif isNullStr(path):
        image_names = []
    
    else:
        raise RuntimeError(f"Wrong type of path. Got {path}, with type {type(path)}.")
    

    return natsort.natsorted(image_names)

def isNullStr(x):
    
    if isinstance(x, list):
        return len(x) == 0
    else:
        return x is None or x == "" or not x


def clear_path(path):

    if os.path.exists(path):
        os.system(f"rm -rf {path}")
        os.makedirs(path)
        
        
def load_config_template(config_template_name, args=None):
    config_template_module = importlib.import_module(f"config.{config_template_name}")
    config_template = getattr(config_template_module, "Config")()

    # remove args with None so that they dont overwrite the configuration with Nones.
    del_attrs = []
    for k, v in vars(args).items():
        if v is None:
            del_attrs.append(k)
    for k in del_attrs:
        delattr(args, k)


    # update config_template with arg_parser object
    if args is not None:
        vars(config_template).update(vars(args))
    
    config_template.update()
    config_template.assert_all_ok()

    return config_template

def path_exists(paths):
    """
    :param paths: path_like str or list of path_like str
    :return: whether given path(s) exists.
    """
    tmp = True
    if isinstance(paths, list) or isinstance(paths, tuple):
        for path in paths:
            tmp = tmp and os.path.exists(path)
    
    elif isinstance(paths, str):
        tmp = tmp and os.path.exists(paths)
        
    return tmp


def printl(*args, **kwargs):
    # print and log
    
    file = kwargs.pop("file")
    print(*args, **kwargs)
    print(*args, **kwargs, file=file)
    
    
