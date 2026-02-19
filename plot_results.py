import pandas as pd, pypsa, os, numpy as np, matplotlib.pyplot as plt, yaml
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.ticker import MultipleLocator



def annuity(lifetime,rate):
    if rate == 0.:
        return 1/lifetime
    else:
        return rate/(1. - 1. / (1. + rate)**lifetime)


#1e3/kW for gas turbine
backup_investment = 1e6
backup_fom = 0.03
backup_lifetime = 25
backup_discount = 0.05
backup_fixed = backup_investment*(annuity(backup_lifetime,backup_discount) + backup_fom)/8760
fuel_cost = 60


def plot_population(pop):

    #vmin = 0
    #vmax = 100
    cmap='RdYlBu_r'

    fig = plt.figure(figsize=(10,6))

    # Create a new figure with a specific map projection
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    
    # Add coastlines and land features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.1)
    
    # Create coordinate arrays
    #lons = np.linspace(-180, 180, int(1+360/config['lon_step']))
    #lats = np.linspace(-90, 90, int(1+180/config['lat_step']))
    
    # Plot the heatmap
    img = ax.pcolormesh(pop.columns+0.5, pop.index+0.5, np.array(pop/1e6),
                       transform=ccrs.PlateCarree(),
                       cmap=cmap#,
                       #vmin=vmin,
                       #vmax=vmax
                       )
    
    # Add colorbar
    plt.colorbar(img, ax=ax, orientation='horizontal', pad=0.05, label='population per pixel [million people]')
    
    # Set title and grid
    ax.set_title(f"population density")
    ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5)

    fig.text(
        0.96, 0.005,
        "© Tom Brown, 2026 - CC BY 4.0 - info & code:\n nworbmot.org/blog/solar-battery-world.html",
        ha="right",
        va="bottom",
        fontsize=8,
        color="0.6"
    )

    fig.tight_layout()
    fig.savefig(f"population_density.pdf")
    fig.savefig(f"population_density.png",dpi=300)


def get_scenario_info(scenario):

    opts = scenario.split("+")
    battcost = None
    wind = False
    for opt in opts:
        if opt[:6] == "tech20":
            year = int(opt[4:8])
        elif opt[:8] == "battcost":
            battcost = int(opt[8:])
        elif opt[:9] == "solarbatt":
            share = int(opt[9:])
        elif opt[:13] == "solarwindbatt":
            wind = True
            share = int(opt[13:])
    return year,share,wind,battcost


def plot_cost_map(scenario):

    vmin = 20
    vmax = 160

    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color='snow')

    fig = plt.figure(figsize=(10,6))


    year,share,wind,battcost = get_scenario_info(scenario)

    if wind:
        nice_name = "solar-wind-battery"
    else:
        nice_name = "solar-battery"

        
    fraction = 1- share/100
    addition = backup_fixed + fraction*fuel_cost

    
    print(scenario,fraction,addition)

    # Create a new figure with a specific map projection
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    
    # Add coastlines and land features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.1)
    
    # Create coordinate arrays
    #lons = np.linspace(-180, 180, int(1+360/config['lon_step']))
    #lats = np.linspace(-90, 90, int(1+180/config['lat_step']))
    
    # Plot the heatmap
    #cost = db["260206-solarbatt2050"]["cost"]["solarbatt95"]
    cost = scenario_db[scenario]["average_cost"].unstack(level=0).reindex(pop.index).reindex(pop.columns,axis=1)#.fillna(-100.)
    img = ax.pcolormesh(cost.columns+0.5, cost.index+0.5, np.array(cost.astype(float)+addition),
                       transform=ccrs.PlateCarree(),
                       cmap=cmap,
                       vmin=vmin,
                       vmax=vmax,
                       )
    
    # Add colorbar
    plt.colorbar(img, ax=ax, orientation='horizontal', pad=0.05, label='average all-in system cost [€/MWh]')
    
    # Set title and grid

    ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5)

    ax.set_title(f"system cost for {share}% utility-scale {nice_name}, {100-share}% fuel in populated areas in {year}")

    fig.text(
        0.96, 0.005,
        "© Tom Brown, 2026 - CC BY 4.0 - info & code:\n nworbmot.org/blog/solar-battery-world.html",
        ha="right",
        va="bottom",
        fontsize=8,
        color="0.6"
    )

    fig.tight_layout()
    fig.savefig(f"cost_map-{scenario}.pdf")
    fig.savefig(f"cost_map-{scenario}.png",dpi=300)


def plot_scenarios(filename):

    fig,ax = plt.subplots(figsize=(6,4))
    thresholds = range(0,161,2)

    shares = [99,95,90]

    if filename == "2030-2050":
        scenario_sel =[f"solarbatt{share}+tech{year}" for year in [2030,2050] for share in shares]
        title = f"solar-battery-fuel system cost for 2030 & 2050 assumptions"
    elif filename in ["2030","2050"]:
        scenario_sel =[f"solarbatt{share}+tech{filename}" for share in shares]
        title = f"solar-battery-fuel system cost in {filename}"
    elif filename == "wind-2030":
        scenario_sel =[f"{tech}{share}+tech2030" for tech in ["solarbatt","solarwindbatt"] for share in shares]
        title = f"solar(-wind)-battery-fuel system cost for {filename[-4:]} assumptions"
    elif filename == "2050-100":
        scenario_sel = ["solarwindbatt100+tech2050","solarwindbatt100+battcost20+tech2050","solarwindbatt100+battcost50+tech2050","solarbatt100+tech2050","solarbatt100+battcost20+tech2050","solarbatt100+battcost50+tech2050"]
        title = f"clean in 2050"
    elif filename == "2050-cheaper_battery":
        scenario_sel = [f"solarbatt{share}{ext}+tech2050" for share in [99] for ext in ["","+battcost50","+battcost20"]]
        title = f"solar-battery-fuel system cost 2050 with cheaper batteries"
    elif filename == "wind-2050-cheaper_battery":
        scenario_sel = [f"solarwindbatt{share}{ext}+tech2050" for share in [99] for ext in ["+battcost50","+battcost20"]]
        title = f"solar-wind-battery-fuel system cost 2050 with cheaper batteries"
    else:
        print(f"filename {filename} unrecognised!")
        sys.exit()


    for scenario in scenario_sel:

        year,share,wind,battcost = get_scenario_info(scenario)

        if wind:
            nice_name = "solar-wind-battery"
        else:
            nice_name = "solar-battery"

        if wind:
            cmap = plt.colormaps['Oranges']
        elif "cheaper" in filename:
            cmap = plt.colormaps['Reds']
        elif not wind and year == 2050:
            cmap = plt.colormaps['Greens']
        else:
            cmap = plt.colormaps['Blues']

        if share == 90:
            i = 0
        elif share == 95:
            i = 1
        else:
            i = 2

        if "cheaper" in filename:
            if "battcost20" in scenario:
                i = 2
            elif "battcost50" in scenario:
                i = 1
            else:
                i = 0

        fraction = 1- float(share)/100

        if fraction == 0:
            addition = 0.
        else:
            addition = backup_fixed + fraction*fuel_cost

        print(scenario,fraction)

        cost_for_truncated = scenario_db[scenario]["average_cost"].copy()
        population = scenario_db[scenario]["population"]

        cost_for_truncated += addition

        cumsum = pd.Series(index=thresholds)

        for th in thresholds:
            #print(th,cost_for_truncated[(cost_for_truncated < th)].shape)
            cumsum.loc[th] = 100*population[cost_for_truncated < th].sum()/population.sum()

        label = f"{int(round(100*(1-fraction)))}% {nice_name}, {int(round(100*fraction))}% fuel"

        if "2030-2050" in filename:
            label = f"{year}: " + label

        if battcost is not None:
            label += f", battery {battcost} €/kWh"

        cumsum_reverse = pd.Series(cumsum.index, cumsum.values)

        cumsum_reverse.plot(ax=ax,ylabel="system cost [€/MWh]", xlabel="share of population [%]",
                              ylim=[20,160],xlim=[0,100],linewidth=3,grid=True,
                              label=label,
                              color=cmap(1-(i+0.5)/3))
    ax.legend()
    ax.set_title(title)

    # Major ticks (as you already have)
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.yaxis.set_major_locator(MultipleLocator(20))

    # Minor ticks
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.yaxis.set_minor_locator(MultipleLocator(10))

    # Grid styling
    ax.grid(which="major", linewidth=1.0, alpha=0.6)
    ax.grid(which="minor", linewidth=0.5, alpha=0.3)

    fig.text(
        0.96, 0.007,
        "© Tom Brown, 2026 - CC BY 4.0\ninfo & code: nworbmot.org/\nblog/solar-battery-world.html",
        ha="right",
        va="bottom",
        fontsize=6,
        color="0.6"
    )

    fig.tight_layout()

    fig.savefig(f"cost_distribution-reverse-{filename}.pdf")
    fig.savefig(f"cost_distribution-reverse-{filename}.png",dpi=300)

def plot_stacked_costs(scenario):

    year,share,wind,battcost = get_scenario_info(scenario)

    data = scenario_db[scenario]
    
    fraction = 1- share/100

    addition = backup_fixed + fraction*fuel_cost

    print(fraction,addition)

    cols = pd.Index([col for col in data.columns if col[-6:] == " totex"])
    totex = data[cols].rename(lambda col: col[:-6],axis=1)
    totex = totex[totex.columns[totex.abs().mean() > 0]].fillna(0.)/8760./100.
    totex["backup fixed cost"] = backup_fixed
    totex["backup fuel"] = fraction*fuel_cost
    totex["total"] = totex.sum(axis=1)
    totex["population"] = data["population"]

    df = totex.sort_values("total")

    df["share"] = df["population"]*100/df["population"].sum()

    # -----------------------
    # Compute bar positions
    # -----------------------
    # left edges so bars touch
    x_left = np.concatenate([[0], np.cumsum(df["share"].values)[:-1]])

    # -----------------------
    # Plot stacked bars
    # -----------------------
    fig, ax = plt.subplots(figsize=(6, 4))

    bottom = np.zeros(len(df))

    cost_cols = totex.columns.difference(pd.Index(["total","population","share"]))

    color = {"solar": "#FFBF00",
             "wind": "royalblue",
             "battery inverter" : "gray",
             "battery storage" : "lightgray",
             "backup fixed cost" : "darkorange",
             "backup fuel" : "brown"}

    for col in cost_cols:
        ax.bar(
            x_left,
            df[col],
            width=df["share"],
            bottom=bottom,
            align="edge",
            label=col,
            color=color[col],
        )
        bottom += df[col].values

    # -----------------------
    # Styling
    # -----------------------

    if wind:
        nice_name = "solar-wind"
    else:
        nice_name = "solar"

    if battcost is not None:
        nice_name += f" battery {battcost} €/kWh"

    ax.set_xlabel("share of population [%]")
    ax.set_ylabel("system cost [€/MWh]")
    ax.set_title(f"system cost breakdown for {share}% {nice_name}-battery, {100-share}% fuel in {year}")


    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1])

    ax.set_xlim(0, df["share"].sum())
    ax.set_ylim(0,160)


    # Major ticks (as you already have)
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.yaxis.set_major_locator(MultipleLocator(20))

    # Minor ticks
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.yaxis.set_minor_locator(MultipleLocator(10))

    # Grid styling
    ax.grid(which="major", linewidth=1.0, alpha=0.6)
    ax.grid(which="minor", linewidth=0.5, alpha=0.3)

    #ax.spines["top"].set_visible(False)
    #ax.spines["right"].set_visible(False)
    #ax.margins(x=0)

    #ax.bar(..., edgecolor="none")

    #line along top
    #ax.plot(x_left + df["share"]/2, df["total"], lw=1.5, color="black", zorder=10)

    ax.grid(True)



    fig.text(
        0.96, 0.007,
        "© Tom Brown, 2026 - CC BY 4.0\ninfo & code: nworbmot.org/\nblog/solar-battery-world.html",
        ha="right",
        va="bottom",
        fontsize=6,
        color="0.6"
    )

    fig.tight_layout()


    filename = f"stacked_cost-{scenario}"

    fig.savefig(f"{filename}.pdf")
    fig.savefig(f"{filename}.png",dpi=300)


if __name__ == "__main__":

    fn = f"config.yaml"
    with open(fn) as f:
        config = yaml.safe_load(f)

    s = pd.read_csv(config['point_list'],index_col=[0,1]).squeeze()

    pop = s.unstack(level=0)

    plot_population(pop)

    results_dir = "results"

    scenarios = [scenario[:-4] for scenario in os.listdir(results_dir) if scenario[-4:] == ".csv"]
    scenario_db = {}

    for scenario in scenarios:
        fn = os.path.join(results_dir,scenario) + ".csv"
        print(scenario)
        scenario_db[scenario] = pd.read_csv(fn,index_col=[0,1])

    for scenario in ["solarbatt90+tech2030"]:
        plot_cost_map(scenario)

    for filename in ["2030","2030-2050","wind-2030","2050-cheaper_battery","wind-2050-cheaper_battery"]:
        plot_scenarios(filename)

    for scenario in ["solarbatt90+tech2030","solarwindbatt99+battcost20+tech2050"]:
        plot_stacked_costs(scenario)
