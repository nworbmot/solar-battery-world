
# calculating the costs of solar-battery-fuel systems for the world

This code calculates the cost of providing a constant demand with
electricity from solar, batteries and a backup fuel generator for any
point in the world, with the option to add wind and change the year
for the technology assumptions.

The results are written up in a [2026 blog post](https://nworbmot.org/blog/solar-battery-world.html).

The model is based on the online tool
 [https://model.energy/](model.energy) and uses [Python for Power
 System Analysis (PyPSA)](https://github.com/PyPSA/PyPSA) for the
 optimisation.



The open weather data comes from the European Centre for Medium-Range
 Weather Forecasts (ECMWF) [ERA5
 dataset](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels)
and is processed using the [atlite
 library](https://github.com/FRESNA/atlite).


# Requirements

## Data

To download the prepared weather data, where the global is split into
eight "octants", run the script "download_octants.py".  This will
store the octants in the folder you should specify in the field
``octant_folder`` in the configuration file ``config.yaml``. The
octants take up 7 GB space.

## Software

PyPSA, xarray, pandas, numpy, shapely

If you do not have access to gurobi, you can change the solver name to
an open-source option like HiGHS or clp.


## License

Copyright 2026 Tom Brown <https://nworbmot.org/>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation; either [version 3 of the
License](LICENSE.txt), or (at your option) any later version.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the [GNU
Affero General Public License](LICENSE.txt) for more details.
