# -*- coding: utf-8 -*-
import os
import shutil

from path import Path


# @Time  : 2023/2/20 3:13
# @Author : boliqq07
# @Software: PyCharm
# @License: MIT License

_dos_help = r"""
复制多个文件夹到新位置。
"""

def _copy_user(i, j, force=False):
    if os.path.isdir(i):
        shutil.copytree(i, j, dirs_exist_ok=force)
    else:
        path_i = os.path.split(j)[0]
        if not os.path.exists(path_i):
            os.makedirs(path_i)
        shutil.copy(i, j)


def copyfile(oldpath, newpath, oldpaths, force=False):

    oldpath = Path(oldpath).abspath()
    newpath = Path(newpath).abspath()

    oldpaths2 = [oldpath.relpathto(Path(i).abspath()) for i in oldpaths]

    newpaths = [Path.joinpath(newpath, i) for i in oldpaths2]

    for i, j in zip(oldpaths, newpaths):
        try:
            _copy_user(i, j, force)
            print(f"copy: {i}.")
        except FileExistsError as e:
            print(f"Error to copy: \n{i} \nto \n{j}.")
            print('You could try use "-force" to cover the exist file.')
            raise e

        except BaseException as e:
            print(e)
            print(f"Error to copy: \n{i} \nto \n{j}.")


class CLICommand:
    __doc__ = _dos_help

    @staticmethod
    def add_arguments(parser):
        parser.add_argument(dest='new_path_arg', nargs="?",  help='new path.', type=str, default=None)
        parser.add_argument('-f', '--path_file', help='source path file.', type=str, default="paths.temp")
        parser.add_argument('-op', '--old_path', help='old path.', type=str, default=".")
        parser.add_argument('-np', '--new_path', help='new path.', type=str, default=".")
        parser.add_argument('-force', '--force', help='disk exist cover.', action="store_true")

    @staticmethod
    def run(args, parser):
        if args.new_path_arg is not None:
            new_path = args.new_path_arg
        else:
            new_path = args.new_path

        if args.old_path == new_path:
            pass
        else:
            with open(args.path_file) as f:
                w = f.readlines()
            if len(w) == 0:
                return
            w = [i.rstrip() for i in w]

            copyfile(args.old_path, new_path, w, args.force)

    @staticmethod
    def parse_args(parser):
        return parser.parse_args()


def main():
    """
    Example:
        $ python this.py -p /home/dir_name
        $ python this.py -f /home/dir_name/path.temp
    """

    from mgetool.cli._formatter import Formatter
    import argparse

    parser = argparse.ArgumentParser(description=_dos_help, formatter_class=Formatter)
    CLICommand.add_arguments(parser=parser)
    args = CLICommand.parse_args(parser=parser)
    CLICommand.run(args=args, parser=parser)


if __name__ == '__main__':
    main()