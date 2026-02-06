configfile: "config.yaml"

import pandas as pd

s = pd.read_csv(config['point_list'],index_col=[0,1])

lonlats = [f"{lon}-{lat}" for (lon,lat), value in s.itertuples() if value > float(config['threshold'])]

wildcard_constraints:
    lonlat="[-0-9]+",
    scenario="[+a-z0-9]+"

rule solve_all:
    input:
        #expand("networks/" + config['run'] + "/{scenario}-{lonlat}.nc",
        expand("csvs/" + config['run'] + "/{scenario}-{lonlat}.csv",
	scenario=config['scenarios'],
	lonlat=lonlats)


rule solve:
    output:
        #network="networks/" + config['run'] + "/{scenario}-{lonlat}.nc",
        csv="csvs/" + config['run'] + "/{scenario}-{lonlat}.csv"
    threads: 4
    resources:
        mem_mb=2000,
        runtime="48h",
    log:
        solver='logs/' + config['run'] + '/{scenario}-{lonlat}_solver.log',
        memory='logs/' + config['run'] + '/{scenario}-{lonlat}_memory.log',
        python='logs/' + config['run'] + '/{scenario}-{lonlat}_python.log',
    script: "solve.py"
