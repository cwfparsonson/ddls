'''
To run this script, must already be in a tmux session and supply the name of this
session with the --session_name flag when calling this script. Script will
then run parallel experiments as windows within this session.
'''
import yaml
import argparse

import subprocess
import sys

import libtmux

import time

if __name__ == '__main__':
    # init arg parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
                '--wandb_sweep_config',
                '-c',
                help='Relative path to <wandb_sweep_config>.yaml file for weights and biases sweep.',
                type=str,
                default='wandb_sweep_config.yaml',
            )
    parser.add_argument(
                '--num_parallel_windows',
                '-n',
                help='Number of runs to conduct in parallel.',
                type=int,
                required=True,
                # default=2,
            )
    parser.add_argument(
                '--session_name',
                '-s',
                help='Name of tmux session in which parallel sweep runs are being conducted.',
                type=str,
                required=True,
                # default='sweep',
            )
    parser.add_argument(
                '--conda_env',
                '-e',
                help='Name of conda environment to activate in each parallel tmux session.',
                type=str,
                default='ddls',
            )
    parser.add_argument(
                '--run_sweep_cmd',
                '-r',
                help='Command to run a pre-defined weights and biases sweep agent. Provide this argument if you have previously generated a sweep command and want to run more sweep agents in parallel. If None, this script will automatically generate a sweep command for you and use it to run parallel sweep agents.',
                type=str,
                default=None,
            )
    parser.add_argument(
                '--run_start_stagger_delay',
                '-d',
                help='Delay (in seconds) with which to stagger parallel runs to decrease chance of conflicting save IDs etc.',
                type=float,
                default=10,
            )
    args = parser.parse_args()

    # run command to generate a weights and biases sweep agent command
    if args.run_sweep_cmd is None:
        gen_sweep_cmd = f'wandb sweep {args.wandb_sweep_config}'.split(' ')
        process = subprocess.Popen(gen_sweep_cmd, bufsize=1, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding='utf-8', errors='replace')
        while True:
            realtime_output = process.stdout.readline()
            if realtime_output == '' and process.poll() is not None:
                break
            if realtime_output:
                # print(f'realtime_output: {realtime_output}')
                if 'Run sweep agent with' in realtime_output.strip():
                    run_sweep_cmd = str(realtime_output.strip()).split('with: ')[-1]
                elif 'View sweep at' in realtime_output.strip():
                    sweep_link = str(realtime_output.strip()).split('at: ')[-1]
                sys.stdout.flush()
    else:
        run_sweep_cmd = args.run_sweep_cmd
        sweep_link = '<unknown>'

    # run parallel sweep agents in separate tmux windows
    print('\n\n\n')
    print(f'~'*100)
    print(f'Launching {args.num_parallel_windows} sweep agents.')
    print(f'View sweep at: {sweep_link}')
    print(f'Launch additional sweep agent(s) with: {run_sweep_cmd}')
    print(f'~'*100)
    server = libtmux.Server()
    session = server.find_where({'session_name': args.session_name})
    for i in range(args.num_parallel_windows):
        start_t = time.time()

        # create window for parallel run
        window = session.new_window(attach=False, window_name=f'{args.session_name}_{i}')

        # get pane of created window
        pane = window.attached_pane

        # activate conda env
        pane.send_keys(f'conda activate {args.conda_env}')

        # run sweep agent
        pane.send_keys(run_sweep_cmd)
        print(f'Launched parallel run {i+1} of {args.num_parallel_windows} in tmux window {window} in {time.time() - start_t:.3f} s.')

        if i != args.num_parallel_windows - 1:
            print(f'Staggering launch of next parallel run by {args.run_start_stagger_delay} seconds...')
            time.sleep(args.run_start_stagger_delay)

    print(f'~'*100)
