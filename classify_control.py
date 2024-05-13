# script for classification experiments
# last maintained: 2023-10-04 17:27:53
# Usage sxsample: $ python classify_control.py cl_conf1.py gpt4 > ./logs/cl_conf1_gpt4.log

import subprocess, time, re, statistics, sys
#
script_name = sys.argv[1]
llm = sys.argv[2]
llm_ = '--llm={:s}'.format(llm)
print('>>>', llm_, flush=True)
#
verbose=False
print('Hello, this is classify_control.py with {llm}'.format(llm=llm), flush=True)

classify_command = ['python', script_name]
descriptions = ['contrast', 'direct', 'direct2', 'none']
seeds = [23, 123, 223, 323, 3407]
wv_dims_list = [-1, 0, 32, 128, 512]

#
total_c = len(descriptions) * len(seeds) * len(wv_dims_list)
c = 0
for desc in descriptions:
    if desc=='none' and llm=='gpt4': continue
    desc_ = '--verb={:s}'.format(desc)
    for dims in wv_dims_list:
        dims_ = '--wv_dim={:s}'.format(str(dims))
        #
        R_per_seeds = []
        for seed in seeds:
            seed_ = '--seed={:s}'.format(str(seed))
            #
            command = classify_command + [desc_, llm_, seed_, dims_, '--verbose=False']
            print('\n>>>>> Executing /',  time.ctime(), ':', ' '.join(command), flush=True)
            R = subprocess.run(command, capture_output=True, text=True)
            try:
                R_stdout_lines = R.stdout.split('\n')
                if verbose: print('---\n', R_stdout_lines[-13:-2], flush=True)
                accuracy_line = R_stdout_lines[-2]
            except:
                print('!!! Retrying this run ...', flush=True)
                continue
            print(accuracy_line, flush=True)
            m = re.search('Accuracy: (\d.\d+)', accuracy_line)
            if not m:
                print('!!! Something wrong happened:', accuracy_line, flush=True)
                break
            accuracy = float(m[1])
            R_per_seeds.append(accuracy)
            c += 1
            print('<<<<< Finished;', 'Run count:', c, 'Total runs:', total_c, '/', time.ctime(), flush=True)
        print('\n', 'llm:', llm, 'desc:', desc, 'wv_dims:', dims, 'seeds:', seeds, 'Ave:', statistics.mean(R_per_seeds), 'Std dev:', statistics.pstdev(R_per_seeds), flush=True)
