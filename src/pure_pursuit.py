import numpy as np

class PurePursuitPlanner:
    """
    Simple Pure Pursuit path follower for 2D (image) coordinates.
    Args:
        waypoints: List of (x, y) tuples representing the path.
        lookahead_distance: Distance ahead to find the target point.
    """
    def __init__(self, waypoints, lookahead_distance=20):
        self.waypoints = np.array(waypoints)
        self.lookahead_distance = lookahead_distance

    def get_steering_point(self, current_position):
        """
        Returns the lookahead (steering) point on the path.
        Args:
            current_position: (x, y) tuple of current position.
        Returns:
            (x, y) tuple of the lookahead point, or None if not found.
        """
        pos = np.array(current_position)
        # Find the first waypoint at least lookahead_distance away
        for wp in self.waypoints:
            if np.linalg.norm(wp - pos) >= self.lookahead_distance:
                return tuple(wp)
        # If none found, return the last waypoint
        if len(self.waypoints) > 0:
            return tuple(self.waypoints[-1])
        return None 