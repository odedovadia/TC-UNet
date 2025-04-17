from .burgers import get_burgers_configs 
from .navier_stokes import get_navier_stokes_configs 
from .wave_equation_2d import get_wave_equation_2d_configs
from .wave_equation_3d import get_wave_equation_3d_configs
from .climate_2d import get_climate_2d_configs
from .climate_3d import get_climate_3d_configs
from .shallow_water import get_shallow_water_configs
from .hypersonics import get_hypersonics_configs
from .hypersonics_time import get_hypersonics_time_configs

BURGERS_CONFIGS = get_burgers_configs()
NAVIER_STOKES_CONFIGS = get_navier_stokes_configs()
WAVE_EQUATION_2D_CONFIGS = get_wave_equation_2d_configs()
WAVE_EQUATION_3D_CONFIGS = get_wave_equation_3d_configs()
CLIMATE_2D_CONFIGS = get_climate_2d_configs()
CLIMATE_3D_CONFIGS = get_climate_3d_configs()
SHALLOW_WATER_CONFIGS = get_shallow_water_configs()
HYPERSONICS_CONFIGS = get_hypersonics_configs()
HYPERSONICS_TIME_CONFIGS  = get_hypersonics_time_configs()
