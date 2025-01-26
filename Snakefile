
wildcard_constraints:
    lon="[0-9]+",
    lat="[0-9]+",
    value="[0-9]+"

rule solve_all:
    input:
        expand("networks/{value}-{lon}-{lat}.nc",
	value=range(50,151,50),
	lon=range(0,360,60),
	lat=range(0,180,60))


rule solve:
    output:
        "networks/{value}-{lon}-{lat}.nc"
    threads: 4
    resources:
        mem_mb=2000
    script: "solve.py"

