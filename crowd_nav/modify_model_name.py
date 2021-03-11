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
    
    exclude = []
    if args.exclude_folder:
        exlude = args.exclude_folder.split(' ')
        
    for out_dir in dir_names:
        if out_dir in exclude:
            continue

        os.chdir(out_dir)
        print('==============================================================')
        print(f'Entering model directory: {out_dir}')

        model_names = glob.glob("rl_model_*.pth")
        model_ckpt = [re.sub(r'rl_model_([0-9]+.*).pth', r'\1', x) for x in model_names]
        if len(model_ckpt) > 0:
            model_ckpt = [float(x) for x in model_ckpt]
            max_ckpt = max(model_ckpt)
        else:
            print('No existing model to change, exiting directory...')
            os.chdir('..')
            continue

        resumed_names = glob.glob("resumed_rl_model_*.pth")
        resumed_ckpt = [re.sub(r'resumed_rl_model_([0-9]+.*).pth', r'\1', x) for x in resumed_names]
        if len(resumed_ckpt) > 0:
            resumed_ckpt = [int(x) for x in resumed_ckpt]
            resumed_ckpt.sort()

            max_c_mod = None
            for c in resumed_ckpt:
                c_mod = (c+1)/resumed_increment + max_ckpt
                c_mod = format_float(c_mod)
                max_c_mod = c_mod
                os.rename(f'resumed_rl_model_{c}.pth', f'rl_model_{c_mod}.pth')
                print(f'Modified file: resumed_rl_model_{c}.pth to rl_model_{c_mod}.pth')
        else:
            print('No resumed model to change')

        model_names = glob.glob("rl_model_*.pth")
        model_ckpt = [re.sub(r'rl_model_([0-9]+.*).pth', r'\1', x) for x in model_names]
        if len(model_ckpt) > 0:
            model_ckpt = [format_float(float(x)) for x in model_ckpt]
            max_ckpt = max(model_ckpt)

            if args.update_training_model:
                try:
                    os.remove('rl_model.pth')
                    print('Removed existing rl_model.pth')
                except FileNotFoundError:
                    pass
                copyfile(f'rl_model_{max_ckpt}.pth', 'rl_model.pth')
                print(f'Updated rl_model.pth file to rl_model_{max_ckpt}.pth')
            
            if args.update_best_val:
                try:
                    os.remove('best_val.pth')
                    print('Removed existing best_val.pth')
                except FileNotFoundError:
                    pass
                copyfile(f'rl_model_{max_ckpt}.pth', 'best_val.pth')
                print(f'Updated best_val.pth file to rl_model_{max_ckpt}.pth')
        else:
            raise Exception('Unhandled error')

        print('Exiting directory...')
        os.chdir('..')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', type=str, default='data/')
    parser.add_argument('-i', '--resumed_increment', type=int, default='1')
    parser.add_argument('-t', '--update_training_model', default=False, action='store_true')
    parser.add_argument('-b', '--update_best_val', default=False, action='store_true')
    parser.add_argument('-e', '--exclude_folder', default=None, type=str)
    
    sys_args = parser.parse_args()
    main(sys_args)

