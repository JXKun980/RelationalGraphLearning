import random

seed_range = [13000, 15000]
seed_cnt = seed_range[1] - seed_range[0]
scenario_2_random = []
scenario_4_random = []
human_cnt = 5

for s in range(*seed_range):
    random.seed(s)
    scenario = []

    for r in range(human_cnt*2):
        scenario.append(random.random())
    if scenario not in scenario_2_random:
        scenario_2_random.append(scenario)

    for r in range(human_cnt*2):
        scenario.append(random.random()) 
    if scenario not in scenario_4_random:
        scenario_4_random.append(scenario)

print(f'Scenario with 2 random numbers repetition count: {seed_cnt-len(scenario_2_random)}, repetition percentage: {(seed_cnt-len(scenario_2_random)) / seed_cnt}')
print(f'Scenario with 4 random numbers repetition count: {seed_cnt-len(scenario_4_random)}, repetition percentage: {(seed_cnt-len(scenario_4_random)) / seed_cnt}')