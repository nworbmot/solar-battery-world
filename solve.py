import pandas as pd, xarray as xr, numpy as np, os, pypsa
from shapely.geometry import box, Point
import matplotlib.pyplot as plt

# following from whobs-server/solve.py

def find_interval(interval_start,interval_length,value):
    return int((value-interval_start)//interval_length)

def get_octant_bounds(quadrant, hemisphere):

    x0 = -180 + quadrant*90.
    x1 = x0 + 90.

    y0 = -90. + hemisphere*90.
    y1 = y0 + 90.

    return x0,x1,y0,y1

def generate_octant_grid_cells(quadrant, hemisphere, mesh=0.5):

    x0,x1,y0,y1 = get_octant_bounds(quadrant, hemisphere)

    x = np.arange(x0,
                  x1 + mesh,
                  mesh)

    y = np.arange(y0,
                  y1 + mesh,
                  mesh)

    #grid_coordinates and grid_cells copied from atlite/cutout.py                                                                                                                      
    xs, ys = np.meshgrid(x,y)
    grid_coordinates = np.asarray((np.ravel(xs), np.ravel(ys))).T

    span = mesh / 2
    return [box(*c) for c in np.hstack((grid_coordinates - span, grid_coordinates + span))]

def get_octant(lon,lat):

    # 0 for lon -180--90, 1 for lon -90-0, etc.                                                                                                                                        
    quadrant = find_interval(-180.,90,lon)

    #0 for lat -90 - 0, 1 for lat 0 - 90                                                                                                                                               
    hemisphere = find_interval(-90,90,lat)

    print(f"octant is in quadrant {quadrant} and hemisphere {hemisphere}")

    rel_x = lon - quadrant*90 + 180.

    rel_y = lat - hemisphere*90 + 90.

    span = 0.5

    n_per_octant = int(90/span +1)

    i = find_interval(0-span/2,span,rel_x)
    j = find_interval(0-span/2,span,rel_y)

    position = j*n_per_octant+i

    print("position",position)

    #paranoid check                                                                                                                                                                    
    if True:
        grid_cells = generate_octant_grid_cells(quadrant, hemisphere, mesh=span)
        assert grid_cells[position].contains(Point(lon,lat))

    return quadrant, hemisphere, position

def pu_from_coordinates(lon,lat,year):

    if lon < -180 or lon > 180 or lat > 90 or lat < -90:
        return "Point's coordinates not within lon*lat range of (-180,180)*(-90,90)", None, None

    quadrant, hemisphere, position = get_octant(lon,lat)

    pu = {}

    for tech in ["solar", "onwind"]:
        filename = os.path.join(snakemake.config['octant_folder'],
				f"octant-{year}-{quadrant}-{hemisphere}-{tech}.nc")
        o = xr.open_dataarray(filename)

        pu[tech] = o.loc[{"dim_0":position}].to_pandas()

    return pd.DataFrame(pu)

defaults = pd.read_csv(snakemake.config['defaults_file'],index_col=[0,1],na_filter=False)

for (n,t) in [("f",float),("i",int)]:
    defaults.loc[defaults["type"] == n, "value"] = defaults.loc[defaults["type"] == n,"value"].astype(t)
#work around fact bool("False") returns True                                                                                                                                           
defaults.loc[defaults.type == "b","value"] = (defaults.loc[defaults.type == "b","value"] == "True")

defaults_t = {str(year): defaults.swaplevel().loc[str(year)] for year in snakemake.config['tech_years']}
defaults_nt = defaults.swaplevel().loc[""]

default_assumptions = pd.concat((defaults_nt,defaults_t[str(snakemake.config['tech_years_default'])])).sort_index()

def annuity(lifetime,rate):
    if rate == 0.:
        return 1/lifetime
    else:
        return rate/(1. - 1. / (1. + rate)**lifetime)

#from elasticity/solve
def solve(assumptions,pu):
    assumptions_df = pd.DataFrame(columns=["FOM","fixed","discount rate","lifetime","investment"],
                                  dtype=float)

    Nyears = len(pu.index.year.unique())

    print(f"{Nyears} years considered")

    techs = [tech[:-5] for tech in assumptions if tech[-5:] == "_cost" and tech[-14:] != "_marginal_cost" and tech != "co2_cost"]

    print(f"calculating costs for {techs}")


    for item in techs:
        assumptions_df.at[item,"discount rate"] = assumptions[item + "_discount"]/100.
        assumptions_df.at[item,"investment"] = assumptions[item + "_cost"]*1e3 if "EUR/kW" in defaults.loc[item + "_cost"]["unit"][0] else assumptions[item + "_cost"]
        assumptions_df.at[item,"FOM"] = assumptions[item + "_fom"]
        assumptions_df.at[item,"lifetime"] = assumptions[item + "_lifetime"]

    assumptions_df["fixed"] = [(annuity(v["lifetime"],v["discount rate"])+v["FOM"]/100.)*v["investment"]*Nyears for i,v in assumptions_df.iterrows()]
    print(assumptions_df)
    
    network = pypsa.Network()

    network.set_snapshots(pu.index)

    network.snapshot_weightings = pd.Series(float(assumptions["frequency"]),index=network.snapshots)

    network.add("Bus","electricity",
                carrier="electricity")

    load = assumptions["load"]

    if assumptions["voll"]:
    	network.add("Generator","load",
                    bus="electricity",
                    carrier="load",
                    marginal_cost=assumptions["voll_price"],
                    p_max_pu=0,
                    p_min_pu=-1,
                    p_nom=assumptions["load"])
    else:
        network.add("Load","load",
                    bus="electricity",
                    carrier="load",
                    p_set=assumptions["load"])


    if assumptions["solar"]:
        network.add("Generator","solar",
                    bus="electricity",
                    carrier="solar",
                    p_max_pu = pu["solar"],
                    p_nom_extendable = True,
                    p_nom_min = assumptions["solar_min"],
                    p_nom_max = assumptions["solar_max"],
                    marginal_cost = 0.1, #Small cost to prefer curtailment to destroying energy in storage, solar curtails before wind                                                 
                    capital_cost = assumptions_df.at['solar','fixed'])

    if assumptions["wind"]:
        network.add("Generator","wind",
                    bus="electricity",
                    carrier="wind",
                    p_max_pu = pu["onwind"],
                    p_nom_extendable = True,
                    p_nom_min = assumptions["wind_min"],
                    p_nom_max = assumptions["wind_max"],
                    marginal_cost = 0.2, #Small cost to prefer curtailment to destroying energy in storage, solar curtails before wind                                                 
                    capital_cost = assumptions_df.at['wind','fixed'])
    
    if assumptions["battery"]:

        network.add("Bus","battery",
                    carrier="battery")

        network.add("Store","battery_energy",
                    bus = "battery",
                    carrier="battery storage",
                    e_nom_extendable = True,
                    e_cyclic=True,
                    capital_cost=assumptions_df.at['battery_energy','fixed'])

        network.add("Link","battery_power",
                    bus0 = "electricity",
                    bus1 = "battery",
                    carrier="battery inverter",
                    efficiency = assumptions["battery_power_efficiency_charging"]/100.,
                    p_nom_extendable = True,
                    capital_cost=assumptions_df.at['battery_power','fixed'])

        network.add("Link","battery_discharge",
                    bus0 = "battery",
                    bus1 = "electricity",
                    carrier="battery discharger",
                    p_nom_extendable = True,
                    efficiency = assumptions["battery_power_efficiency_discharging"]/100.)

    network.add("Bus",
                "chemical",
                carrier="chemical")


    network.add("Link",
                "hydrogen_turbine",
                bus0="chemical",
                bus1="electricity",
                carrier="hydrogen turbine",
                p_nom_extendable=True,
                efficiency=assumptions["hydrogen_turbine_efficiency"]/100.,
        		capital_cost=assumptions_df.at["hydrogen_turbine","fixed"]*assumptions["hydrogen_turbine_efficiency"]/100.)  #NB: fixed cost is per MWel 
    
    network.add("Link",
                    "methanol synthesis",
                    bus0="electricity",
                    bus1="chemical",
                    carrier="methanol synthesis",
                    p_nom_extendable=True,
                    efficiency=0.5,
                    capital_cost=assumptions_df.at["hydrogen_electrolyser","fixed"]+assumptions_df.at["methanolisation","fixed"]*assumptions["methanolisation_efficiency"]) #NB: cost is EUR/kW_MeOH                                                                                                                                                                         

    network.add("Store",
                    "methanol",
                    bus="chemical",
                    carrier="methanol storage",
                    e_nom_extendable=True,
                    e_cyclic=True,
                    capital_cost=assumptions_df.at["liquid_carbonaceous_storage","fixed"]/4.4)
    
    
    network.consistency_check()

    solver_name = "gurobi"
    solver_options = {"threads": 4,
    "method": 2, # barrier                                                                                                                                                                
    "crossover": 0,
    "BarConvTol": 1.e-8,
                     }

    network.optimize.create_model()

    if assumptions["battery"]:
    	network.model.add_constraints(network.model["Link-p_nom"].loc["battery_power"]
                                      -network.links.loc["battery_discharge", "efficiency"]*
                                      network.model["Link-p_nom"].loc["battery_discharge"] == 0,
                                      name='charger_ratio')


    status, termination_condition = network.optimize.solve_model(solver_name=solver_name,
                                                                 solver_options=solver_options)

        
    return network



if __name__ == "__main__":

    if 'snakemake' not in globals():
        from pypsa.descriptors import Dict
        import yaml
        from types import SimpleNamespace

        snakemake = SimpleNamespace()

        with open('config.yaml') as f:
            snakemake.config = yaml.safe_load(f)

        snakemake.wildcards = Dict({"value" : 50,
                                    "lon" : 10,
                                    "lat" : 50,
                                    })

        snakemake.output = ["networks/{}-{}-{}.nc".format(snakemake.wildcards.value,
                                                          snakemake.wildcards.lon,
                                                          snakemake.wildcards.lat,
                                                          )]
        snakemake.log = {
            "python": f"logs/{snakemake.wildcards['country']}-{snakemake.wildcards['scenario']}-python.log",
            "solver": f"logs/{snakemake.wildcards['country']}-{snakemake.wildcards['scenario']}-solver.log",
            }

    year = 2011

    lon = int(snakemake.wildcards.lon)

    if lon >= 180:
        lon -= 360

    lat = int(snakemake.wildcards.lat)

    if lat >= 90:
        lat -= 180

    pu = pu_from_coordinates(lon,
                             lat,
                             year)

    assumptions = default_assumptions["value"].to_dict()
    assumptions["frequency"] = 1.

    scenario = snakemake.wildcards.scenario

    opts = scenario.split("+")

    for opt in opts:
        if opt == "vanilla":
            continue
        elif opt == "nowind":
            assumptions["wind"] = False
        elif opt[:4] == "voll":
            assumptions["voll"] = True
            assumptions["voll_price"] = int(opt[4:])

    n = solve(assumptions,pu)

    n.export_to_netcdf(snakemake.output[0],
                       # compression of network
                       float32=True, compression={'zlib': True, "complevel":9, "least_significant_digit":5}
                       )
