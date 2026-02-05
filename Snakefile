configfile: "config.yaml"

import pandas as pd

s = pd.read_csv(config['point_list'],index_col=[0,1])

lonlats = [f"{lon}-{lat}" for (lon,lat), value in s.itertuples() if value > float(config['threshold'])]

wildcard_constraints:
    lonlat="[-0-9]+",
    scenario="[+a-z0-9]+"

rule solve_all:
    input:
        expand("networks/" + config['run'] + "/{scenario}-{lonlat}.nc",
	scenario=config['scenarios'],
	lonlat=lonlats)


rule solve:
    output:
        "networks/" + config['run'] + "/{scenario}-{lonlat}.nc"
    threads: 4
    resources:
        mem_mb=2000
    script: "solve.py"
