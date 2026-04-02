import numpy as np

def unicycle_vertical_strip_generator(
    x_value, ybound, theta_value, num_u, random_order=True
):
    """
    Generate a vertical sequence of points with same x and theta,
    only changing y monotonically (up or down).
    Output format matches unicycle_single_int_trajectory_generator:
      (x0, y0), (x1, y1), (orientation0, orientation1), (v, w)

    Args:
        x_value: float — fixed x coordinate
        ybound: (ymin, ymax)
        theta_value: float — fixed heading (radians)
        num_u: int — number of pseudo-steps
        random_order: bool — randomly choose "up" or "down" (default True)
    """
    yl, yh = ybound
    ys = np.sort(np.random.uniform(yl, yh, num_u + 1))
    
    if random_order:
        direction = np.random.choice(["up", "down"])
    else:
        direction = "up"
    if direction == "down":
        ys = ys[::-1]

    traj = []
    for i in range(num_u):
        y0 = ys[i]
        y1 = ys[i + 1]
        x0 = x1 = x_value
        theta0 = theta1 = theta_value
        v = normalize_v(abs(y1 - y0), ybound)
        w = 0.0
        traj.append(((x0, y0), (x1, y1), (theta0, theta1), (v, w)))

    return np.array(traj, dtype=object)

def normalize_v(v,bound):
    low = bound[0]
    high = bound[1]
    length = high-low

    return 2*v/length 

def unicycle_single_int_trajectory_generator(
    x0y0_pair, xbound, ybound, num_u,
    dt=0.1, min_step=3, max_step=10, max_angle_step_deg=30
):
    """
    Returns list of tuples:
      (x0_int, y0_int), (x1_int, y1_int), (orientation0_int, orientation1_int), (v, w)
    with length=num_u
    """
    xl, xh = xbound
    yl, yh = ybound

    # Start state (float)
    x0, y0 = x0y0_pair
    x0 = float(x0)
    y0 = float(y0)
    orientation0 = np.random.uniform(np.pi/2, 3/2*np.pi)  # Random initial orientation in radians
    # orientation0 = np.random.uniform(0, 2*np.pi)  # Random initial orientation in radians
    max_angle_step_deg = max_angle_step_deg /dt

    thetabound = [np.pi/2, 3/2*np.pi]
    # thetabound = [0, 2*np.pi]

    traj = []

    strip = unicycle_vertical_strip_generator(x0, ybound, orientation0, num_u, random_order=True)

    return strip


if __name__ == "__main__":    # Test
    traj = unicycle_single_int_trajectory_generator(
        (50, 50), (0, 100), (0, 100), 5
    )
    #plot the trajectory using arrows
    import matplotlib.pyplot as plt
    x = [step[0][0] for step in traj] + [traj[-1][1][0]]
    y = [step[0][1] for step in traj] + [traj[-1][1][1]]
    plt.figure()
    plt.plot(x, y, marker='o')
    arrow_len = 6.0
    for i in range(len(traj)):
        x0, y0 = traj[i][0]
        # orientation stored as (theta0, theta1) in traj[i][2]
        theta0 = traj[i][2][0]
        # choose an arrow length (tweak as needed)+        arrow_len = 6.0
        dx = arrow_len * np.cos(theta0)
        dy = arrow_len * np.sin(theta0)
        plt.arrow(x0, y0, dx, dy,
                  head_width=1.5, head_length=2.0, length_includes_head=True,
                  fc='red', ec='red', linewidth=1.0)
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.title("Unicycle Vertical Strip Trajectory")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid()
    plt.show()