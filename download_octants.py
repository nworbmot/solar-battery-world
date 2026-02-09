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

from urllib.request import urlretrieve

import yaml, os

with open('config.yaml') as f:
    config = yaml.safe_load(f)


weather_years = [2011]

for weather_year in weather_years:
    for quadrant in range(4):
        for hemisphere in range(2):
            for tech in ["solar","onwind"]:
                octant = f"octant-{weather_year}-{quadrant}-{hemisphere}-{tech}.nc"
                url = f"https://model.energy/octants/{octant}"
                filename = os.path.join(config['octant_folder'],
                                        octant)
                urlretrieve(url,
                            filename)

                                        

