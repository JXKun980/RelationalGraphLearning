scen_cnt = [7,6,10,1]
data_file = 'data.txt'
fail_file = 'fail.txt'
output_file = 'out.txt'

with open(data_file, 'r') as fdata, open(fail_file, 'r') as ffail, open(output_file, 'w') as fout:
    failed_cases = [int(x) for x in ffail.readline().split(' ')]
    fdata.readline()

    print(scen_cnt)

    cur_case = 0
    while (sum(scen_cnt) != 0) and (cur_case < 2000):
        line = fdata.readline()
        while cur_case in failed_cases:
            cur_case += 1
        if scen_cnt[cur_case % 4] > 0:
            fout.write(line)
            scen_cnt[cur_case%4] -= 1
        print(f'{scen_cnt}, {cur_case}')
        cur_case += 1