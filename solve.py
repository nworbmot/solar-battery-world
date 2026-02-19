## Copyright 2026 Tom Brown

## This program is free software; you can redistribute it and/or
## modify it under the terms of the GNU Affero General Public License as
## published by the Free Software Foundation; either version 3 of the
## License, or (at your option) any later version.

## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU Affero General Public License for more details.

## License and more information at:
## https://github.com/nworbmot/solar-battery-world

import pandas as pd, xarray as xr, numpy as np, os, pypsa, sys
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

def annuity(lifetime,rate):
    if rate == 0.:
        return 1/lifetime
    else:
        return rate/(1. - 1. / (1. + rate)**lifetime)



def generate_overview(network):

    results_overview = n.buses_t.marginal_price.mean().rename(lambda n : f"{n} average price")
    results_overview.index.name = ""

    if "load" in network.loads.index:
        total_load = network.loads.at["load","p_set"]
    else:
        total_load = network.generators_t["p"]["load"].mean()

    stats = network.statistics().groupby(level=1).sum()

    stats["Total Expenditure"] = stats[["Capital Expenditure","Operational Expenditure"]].sum(axis=1)

    for name,full_name in [("capex","Capital Expenditure"),("opex","Operational Expenditure"),("totex","Total Expenditure"),("capacity","Optimal Capacity")]:
        results_overview = pd.concat((results_overview,
                                      stats[full_name].rename(lambda x: x+ f" {name}")))

    results_overview["average_cost"] = sum([results_overview[s] for s in results_overview.index if s[-6:] == " totex"])/total_load/8760.

    #report capacity from p1 not p0
    if "hydrogen turbine capacity" in results_overview:
        results_overview.loc["hydrogen turbine capacity"] *= network.links.at["hydrogen_turbine","efficiency"]


    results_overview = pd.concat((results_overview,
                                  (stats["Curtailment"]/(stats["Supply"]+stats["Curtailment"])).rename(lambda x: x+ " curtailment")))

    results_overview = pd.concat((results_overview,
                                  (stats["Total Expenditure"]/(stats["Supply"])).rename(lambda x: x+ " LCOE")))

    results_overview = pd.concat((results_overview,
                                  stats["Capacity Factor"].rename(lambda x: x+ " cf used")))

    results_overview = pd.concat((results_overview,
                                  ((stats["Supply"]+stats["Curtailment"])/stats["Optimal Capacity"]/network.snapshot_weightings["generators"].sum()).rename(lambda x: x+ " cf available")))

    #RMV
    bus_map = (network.buses.carrier == "electricity")
    bus_map.at[""] = False
    for c in network.iterate_components(network.one_port_components):
        items = c.df.index[c.df.bus.map(bus_map).fillna(False)]
        if len(items) == 0:
            continue
        rmv = (c.pnl.p[items].multiply(network.buses_t.marginal_price["electricity"], axis=0).sum()/c.pnl.p[items].sum()).groupby(c.df.loc[items,'carrier']).mean()/results_overview["electricity average price"]
        results_overview = pd.concat((results_overview,
                                      rmv.rename(lambda x: x+ " rmv").replace([np.inf, -np.inf], np.nan).dropna()))

    for c in network.iterate_components(network.branch_components):
        for end in [col[3:] for col in c.df.columns if col[:3] == "bus"]:
            items = c.df.index[c.df["bus" + str(end)].map(bus_map,na_action=None)]
            if len(items) == 0:
                continue
            if c.pnl["p"+end].empty:
                continue
            rmv = (c.pnl["p"+end][items].multiply(network.buses_t.marginal_price["electricity"], axis=0).sum()/c.pnl["p"+end][items].sum()).groupby(c.df.loc[items,'carrier']).mean()/results_overview["electricity average price"]
            results_overview = pd.concat((results_overview,
                                          rmv.rename(lambda x: x+ " rmv").replace([np.inf, -np.inf], np.nan).dropna()))

    return results_overview

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

    if assumptions["hydrogen"]:
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

    if assumptions["fossil"]:
        network.add("Generator",
                    "fossil",
                    bus="chemical",
                    carrier="fossil",
                    marginal_cost=assumptions["fossil_price"],
                    p_nom_extendable=True)

    for i in range(1,3):
        name = "dispatchable" + str(i)
        if assumptions[name]:
            network.add("Carrier",name,
                        co2_emissions=assumptions[name+"_emissions"])
            network.add("Generator",name,
                        bus="electricity",
                        carrier=name,
                        p_nom_extendable=True,
                        marginal_cost=assumptions[name+"_marginal_cost"],
                        capital_cost=assumptions_df.at[name,'fixed'])

    if assumptions["co2_limit"]:
        network.add("GlobalConstraint","co2_limit",
                    sense="<=",
                    constant=assumptions["co2_emissions"]*assumptions["load"]*network.snapshot_weightings.objective.sum())

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

    if assumptions["maxcost"]:

        objective = 0.
        for component in ["Generator","Link","Store"]:
            cost = network.static(component)["capital_cost"]
            cost.index.name = f"{component}-ext"
            if component == "Store":
                attr = "e"
            else:
                attr = "p"
            objective += (network.model[f"{component}-{attr}_nom"]*cost).sum()

        network.model.add_constraints(objective == assumptions["maxcost_level"]*assumptions["load"]*8760,
                                      name='maxcost')

    status, termination_condition = network.optimize.solve_model(solver_name=solver_name,
                                                                 log_fn=snakemake.log.solver,
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

    lonlat = snakemake.wildcards.lonlat
    lon = int(lonlat[:lonlat[1:].find("-")+1])
    lat = int(lonlat[lonlat[1:].find("-")+2:])

    print(f"{lonlat} -> {lon},{lat}")

    if lon >= 180:
        lon -= 360

    if lat >= 90:
        lat -= 180

    pu = pu_from_coordinates(lon,
                             lat,
                             year)


    scenario = snakemake.wildcards.scenario

    if 'tech20' in scenario:
        tech_year = scenario[scenario.find('tech20')+4:scenario.find('tech20')+8]
    else:
        tech_year = str(snakemake.config['tech_years_default'])

    print(f"Using technology assumptions for year {tech_year}")

    default_assumptions = pd.concat((defaults_nt,defaults_t[tech_year])).sort_index()

    assumptions = default_assumptions["value"].to_dict()
    assumptions["frequency"] = 1.
    assumptions["fossil"] = False
    assumptions["fossil_price"] = 50.
    assumptions["maxcost"] = False
    assumptions["maxcost_level"] = 0.

    opts = scenario.split("+")

    for opt in opts:
        if opt == "vanilla":
            continue
        elif opt[:6] == "tech20":
            continue
        elif opt == "nowind":
            assumptions["wind"] = False
        elif opt[:4] == "voll":
            assumptions["voll"] = True
            assumptions["voll_price"] = int(opt[4:])
        elif opt[:6] == "fossil":
            assumptions["fossil"] = True
            assumptions["fossil_price"] = float(opt[6:])
        elif opt[:9] == "solarbatt":
            assumptions["hydrogen"] = False
            assumptions["wind"] = False
            assumptions["dispatchable1"] = True
            assumptions["dispatchable1_cost"] = 0.
            assumptions["dispatchable1_emissions"] = 1000.
            assumptions["dispatchable1_marginal_cost"] = 0.
            assumptions["co2_limit"] = True
            co2_emissions = (100-float(opt[9:]))*10
            print(f"based on {opt[9:]} setting, restricting emissions to {co2_emissions}")
            assumptions["co2_emissions"] = co2_emissions
        elif opt[:13] == "solarwindbatt":
            assumptions["hydrogen"] = False
            assumptions["dispatchable1"] = True
            assumptions["dispatchable1_cost"] = 0.
            assumptions["dispatchable1_emissions"] = 1000.
            assumptions["dispatchable1_marginal_cost"] = 0.
            assumptions["co2_limit"] = True
            co2_emissions = (100-float(opt[13:]))*10
            print(f"based on {opt[13:]} setting, restricting emissions to {co2_emissions}")
            assumptions["co2_emissions"] = co2_emissions
        elif opt[:8] == "battcost":
            assumptions["battery_energy_cost"] = float(opt[8:])
            print(f"battery energy cost changed to {assumptions['battery_energy_cost']}")
        elif opt[:7] == "maxcost":
            assumptions["hydrogen"] = False
            assumptions["wind"] = False
            assumptions["maxcost"] = True
            assumptions["maxcost_level"] = float(opt[7:])
            print(f"setting a maximum system cost to {assumptions['maxcost_level']} €/MWh")
        else:
            print(f"option {opt} not recognised, quitting")
            sys.exit()

    n = solve(assumptions,pu)

    results_overview = generate_overview(n)

    results_overview.to_csv(snakemake.output.csv)

    #n.export_to_netcdf(snakemake.output[0],
    #                   # compression of network
    #                   float32=True, compression={'zlib': True, "complevel":9, "least_significant_digit":5}
    #                   )
