import numpy as np

# no quantization, but normalize to [-1,1]

def normalize_pos(pos,bound):
    low = bound[0]
    high = bound[1]
    length = high-low

    return 2*(pos-low)/length - 1

def normalize_v(v,bound):
    low = bound[0]
    high = bound[1]
    length = high-low

    return 2*v/length 

# def quantize_angle_deg(theta):
#     # Quantize to nearest even integer degree [0, 360)
#     return int(np.round(theta / 2.0) * 2) % 360

# def quantize_pos(val):
#     # Quantize to nearest integer pixel
#     return int(np.round(val))

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
    #orientation0 = np.random.uniform(0, np.pi)
    #orientation0 = np.random.uniform(0, 2*np.pi)  # Random initial orientation in radians
    max_angle_step_deg = max_angle_step_deg /dt

    thetabound = [np.pi/2, 3/2*np.pi]
    #thetabound = [0, np.pi]
    #thetabound = [0, 2*np.pi]

    traj = []

    for _ in range(num_u):
        # Try up to N times to get a valid step (stays in bounds and step size/turn size valid)
        for _attempt in range(50):
            step_size = np.random.uniform(min_step, max_step)
            max_w = np.deg2rad(max_angle_step_deg)
            w = np.random.uniform(-max_w, max_w)
            v = step_size / dt
            orientation1 = orientation0 + w * dt

            x1 = x0 + v * np.cos(orientation0) * dt
            y1 = y0 + v * np.sin(orientation0) * dt


            # Check bounds
            if (xl <= x1 <= xh) and (yl <= y1 <= yh) and (thetabound[0] <= (orientation1%(2*np.pi)) <= thetabound[1]):
                break

        else:
            # If after 50 tries you can't get a valid step, stop trajectory early
            break

        orientation1 = orientation1 % (2*np.pi)
        v_n = normalize_v(v,xbound)

        traj.append((
            (x0, y0),
            (x1, y1),
            (orientation0, orientation1),
            (v_n, w)
        ))

        # Move to next step (float)
        x0, y0, orientation0 = x1, y1, orientation1

    return np.array(traj, dtype=object)










