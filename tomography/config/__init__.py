try:
	from .prototype import POINT_FIELDS_XYZI, GRID_POINTS_XYZI
except ModuleNotFoundError as error:
	if error.name != 'sensor_msgs':
		raise

from .param import Config

from .scene_spiral import SceneSpiral
from .scene_building import SceneBuilding
from .scene_plaza import ScenePlaza