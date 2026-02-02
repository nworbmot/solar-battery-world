configfile: "config.yaml"

wildcard_constraints:
    lon="[0-9]+",
    lat="[0-9]+",
    value="[0-9]+"

rule solve_all:
    input:
        expand("networks/{value}-{lon}-{lat}.nc",
	value=range(config['value_start'],config['value_stop'],config['value_step']),
	lon=range(0,360,config['lon_step']),
	lat=range(0,180,config['lat_step']))


rule solve:
    output:
        "networks/{value}-{lon}-{lat}.nc"
    threads: 4
    resources:
        mem_mb=2000
    script: "solve.py"

