# undefined behaviour of jupyter server cannot start from project root
# temporary fix by manual change of dir when detecting starting wrong one
# with fixed jupyter notebook all path files can be also fixed to use relative one
# which enables default compatibility with colab
# colab also has a problem with jupyter server, but it is not fixed yet
# @TODO investigate further

import os
import sys


def fix_jupyter_path():
    in_colab = 'google.colab' in sys.modules

    if in_colab:
        raise Exception('Colab is not supported yet')
    else:
        if os.name == 'nt': # windows
            if os.getcwd()[-14:] == '\\src\\notebooks':
                os.chdir(os.getcwd()[:-14])
        elif os.name == 'posix': # linux
            if os.getcwd()[-14:] == '/src/notebooks':
                os.chdir(os.getcwd()[:-14])
        else:
            raise Exception('OS not supported')
