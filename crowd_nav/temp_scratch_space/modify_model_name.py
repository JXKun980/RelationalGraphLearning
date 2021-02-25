import os
import glob
import re
import argparse
from shutil import copyfile

def format_float(n):
    if n % 1:
        return n
    else:
        return int(n)

def main(args):
    resumed_increment = args.resumed_increment
    data_dir = args.data_dir

    os.chdir(data_dir)
    dir_names = glob.glob('output_*')

    for out_dir in dir_names:
        os.chdir(out_dir)

        model_names = glob.glob("rl_model_*.py")
        model_ckpt = [re.sub(r'rl_model_([0-9]+).py', r'\1', x) for x in model_names]
        if len(model_ckpt) > 0:
            model_ckpt = [int(x) for x in model_ckpt]
            max_ckpt = max(model_ckpt)
        else:
            continue

        resumed_names = glob.glob("resumed_rl_model_*.py")
        resumed_ckpt = [re.sub(r'resumed_rl_model_([0-9]+).py', r'\1', x) for x in resumed_names]
        if len(resumed_ckpt) > 0:
            resumed_ckpt = [int(x) for x in resumed_ckpt]
            resumed_ckpt.sort()

            max_c_mod = None
            for c in resumed_ckpt:
                c_mod = (c+1)/resumed_increment + max_ckpt
                c_mod = format_float(c_mod)
                max_c_mod = c_mod
                os.rename(f'resumed_rl_model_{c}.py', f'rl_model_{c_mod}.py')

            if args.update_training_model:
                try:
                    os.remove('rl_model.py')
                except FileNotFoundError:
                    pass
                copyfile(f'rl_model_{max_c_mod}.py', 'rl_model.py')
            
            if args.update_best_val:
                try:
                    os.remove('best_val.py')
                except FileNotFoundError:
                    pass
                copyfile(f'rl_model_{max_c_mod}.py', 'bset_val.py')
        else:
            continue
        
        os.chdir('..')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', type=str, default='data/')
    parser.add_argument('-i', '--resumed_increment', type=int, default='1')
    parser.add_argument('-t', '--update_training_model', default=False, action='store_true')
    parser.add_argument('-b', '--update_best_val', default=False, action='store_true')
    
    sys_args = parser.parse_args()
    main(sys_args)

