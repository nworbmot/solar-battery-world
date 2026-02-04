configfile: "config.yaml"

wildcard_constraints:
    lon="[0-9]+",
    lat="[0-9]+",
    scenario="[+a-z0-9]+"

rule solve_all:
    input:
        expand("networks/" + config['run'] + "/{scenario}-{lon}-{lat}.nc",
	scenario=config['scenarios'],
	lon=range(0,360,config['lon_step']),
	lat=range(0,180,config['lat_step']))


rule solve:
    output:
        "networks/" + config['run'] + "/{scenario}-{lon}-{lat}.nc"
    threads: 4
    resources:
        mem_mb=2000
    script: "solve.py"
