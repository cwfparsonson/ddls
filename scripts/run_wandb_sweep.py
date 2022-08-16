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
                default=4,
            )
    parser.add_argument(
                '--session_name',
                '-s',
                help='Name of sweep run to prefix tmux sessions with.',
                type=str,
                default='sweep',
            )
    parser.add_argument(
                '--conda_env',
                '-e',
                help='Name of conda environment to activate in each parallel tmux session.',
                type=str,
                default='ddls',
            )
    args = parser.parse_args()

    # run command to generate a weights and biases sweep agent command
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

    # run parallel sweep agents in separate tmux windows
    print(f'Launching {args.num_parallel_windows} sweep agents. View sweep at: {sweep_link}')
    server = libtmux.Server()
    session = server.find_where({'session_name': args.session_name})
    for i in range(args.num_parallel_windows):
        # create window for parallel run
        window = session.new_window(attach=False, window_name=f'{args.session_name}_{i}')

        # get pane of created window
        pane = window.attached_pane

        # activate conda env
        pane.send_keys(f'conda activate {args.conda_env}')

        # run sweep agent
        pane.send_keys(run_sweep_cmd)
        print(f'Launched parallel run {i+1} of {args.num_parallel_windows} in tmux window {window}')
