import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)
print("LRP print this_dir", this_dir)

# Add lib to PYTHONPATH
lib_path = osp.join(this_dir, '..', 'lib_fr')
add_path(lib_path)
print("LRP lib path:", lib_path)

coco_path = osp.join(this_dir, '..', 'data', 'coco', 'PythonAPI')
add_path(coco_path)
