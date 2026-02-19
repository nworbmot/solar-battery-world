import pandas as pd, yaml, os

fn = f"config.yaml"
with open(fn) as f:
    config = yaml.safe_load(f)

s = pd.read_csv(config['point_list'],index_col=[0,1]).squeeze()

runs =  ["260206-solarbatt","260206-solarbatt2050","260207-solarwindbatt","260213-cheaper_battery","260214-cheaper_battery-wind","260215-2050clean"]

db = {}

for run in runs:
    db[run] = {}
    fn = f"csvs/{run}/config.yaml"
    with open(fn) as f:
        db[run]["config"] = yaml.safe_load(f)
    threshold = float(db[run]["config"]["threshold"])
    truncated = s[s>threshold]
    print(f"{run}: threshold {threshold} with {len(truncated)} entries")
    default_year = db[run]["config"]['tech_years_default']
    print(default_year)

    db[run]["data"] = {}
    for scenario in db[run]["config"]["scenarios"]:
        # to seed columns, use a node with high population and wind even with battcost20
        db[run]["data"][scenario] = pd.DataFrame(index=truncated.index,
                                                 columns=pd.read_csv(f"csvs/{run}/{scenario}-31-30.csv",index_col=0).squeeze().index)
        for (lon,lat), value in s.items():
            if value < threshold:
                continue
            fn = f"csvs/{run}/{scenario}-{lon}-{lat}.csv"
            if not os.path.isfile(fn):
                print(f"warning, didn't find {fn}")
                continue
            data = pd.read_csv(fn,index_col=0).squeeze()
            db[run]["data"][scenario].loc[lon,lat] = data
        db[run]["data"][scenario]["population"] = truncated

        scenario_new = scenario
        if "tech20" not in scenario:
            scenario_new = f"{scenario}+tech{default_year}"
        print(scenario_new)
        db[run]["data"][scenario].to_csv(f"results/{scenario_new}.csv")
