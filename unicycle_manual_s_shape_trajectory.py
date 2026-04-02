import numpy as np
import casadi as ca

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

    v_min = min_step/dt
    v_max = max_step/dt
    # Start state (float)
    x0, y0 = x0y0_pair
    x0 = float(x0)
    y0 = float(y0)
    orientation0 = np.random.uniform(np.pi/2, 3/2*np.pi)  # Random initial orientation in radians
    
    if (x0-15)<=xl:
        return np.array([])

    xT = np.random.uniform(x0-15, x0-12)
    yT = np.random.uniform(yl, yh)
    orientationT = orientation0
    max_angle_step_deg = max_angle_step_deg*2 /dt
    w_max = np.deg2rad(max_angle_step_deg)

    thetabound = [np.pi/2, 3/2*np.pi]





    x_traj, y_traj, th_traj, v_traj, w_traj = unicycle_S_shape_solver_sharp_turns(
        x0, y0, orientation0,
        xT, yT, orientationT,
        N=num_u-1, dt=0.1,v_min=v_min,
        v_max=v_max, w_max=w_max,
        thetabound= thetabound
    )

    if len(x_traj) ==0:
        return np.array([])
    
    traj = []
    for i in range(len(x_traj)-1):
        x0 = x_traj[i]
        y0 = y_traj[i]
        orientation0 = th_traj[i]
        x1 = x_traj[i+1]
        y1 = y_traj[i+1]
        orientation1 = th_traj[i+1]
        v = v_traj[i]
        w = w_traj[i]

        v_n = normalize_v(v,xbound)

        traj.append((
            (x0, y0),
            (x1, y1),
            (orientation0, orientation1),
            (v_n, w)
        ))

    # apply a last step of constant v and zero w
    v_last = np.random.uniform(v_min, v_max)
    v_last_n = normalize_v(v_last,xbound)
    orientation_last = th_traj[-1]
    x_last = x_traj[-1] + v_last * np.cos(orientation_last) * dt
    y_last = y_traj[-1] + v_last * np.sin(orientation_last) * dt
    traj.append((
        (x_traj[-1], y_traj[-1]),
        (x_last, y_last),
        (orientation_last, orientation_last),
        (v_last_n, 0)
    ))

    return np.array(traj, dtype=object)









def unicycle_S_shape_solver_sharp_turns(
    x0, y0, th0,
    xT, yT, thT,
    N=40, dt=0.1,
    v_min=0.1, v_max=5.0, w_max=np.pi/3/0.1,
    thetabound=(0.5*np.pi, 1.5*np.pi),xbound = (35,102), ybound = (35,90)
):
    """
    Solve for a unicycle trajectory connecting start to goal,
    forward-only (v>0), leftward facing, and minimizing angle change dispersion.
    Returns arrays of x, y, theta, v, w.
    """
    # print("Solving unicycle S-shape trajectory from ({:.2f},{:.2f},{:.2f}) to ({:.2f},{:.2f},{:.2f})".format(
    #     x0, y0, th0, xT, yT, thT
    # ))
    # print("Velocity bounds: [{:.2f}, {:.2f}], Angular velocity bound: {:.2f} rad/s".format(
    #     v_min, v_max, w_max
    # ))  
    # print("Position bounds: x[{:.2f}, {:.2f}], y[{:.2f}, {:.2f}], Orientation bounds: [{:.2f}, {:.2f}]".format(
    #     xbound[0], xbound[1], ybound[0], ybound[1], thetabound[0], thetabound[1]
    # ))
    # print("Number of steps: {}, dt: {:.2f}s".format(N, dt))

    x_min, x_max = xbound
    y_min, y_max = ybound

    opti = ca.Opti()

    # Decision variables
    x = opti.variable(N+1)
    y = opti.variable(N+1)
    th = opti.variable(N+1)
    v = opti.variable(N)
    w = opti.variable(N)

    # Objective: minimize smooth spread of turning (favor piecewise straight)
    obj = 0

    # Basic effort penalty
    obj += 0.05 * ca.sumsqr(v) + 0.01 * ca.sumsqr(w)

    # Penalize *gradual* angular changes (we want concentrated sharp turns)
    # obj += 10.0 * ca.sumsqr(ca.diff(w))   # smaller diff(w) → smoother heading change dispersion
    # but we want opposite: penalize small w changes? actually we want fewer large changes
    # Instead of penalizing diff(w)^2, we can *reward* sparsity via L1, but L1 is nondiff
    # So approximate with inverted quadratic penalty:
    obj -= 0.2 * ca.sumsqr(w)             # encourages use of strong turns sparsely

    # epsilon = 1e-3
    # lam_sparse = 1.0
    # obj += lam_sparse * ca.sumsqr(ca.sqrt(w**2 + epsilon))

    alpha = 5.0    # slope of tanh approximation
    sign_approx = ca.tanh(alpha * w)
    gamma = 10.0    # weight for sign change penalty
    obj += gamma * ca.sumsqr(ca.diff(sign_approx))

    penalty_weight = 100.0
    obj += penalty_weight * ca.sumsqr(ca.fmax(0, x - x_max))
    obj += penalty_weight * ca.sumsqr(ca.fmax(0, x_min - x))
    obj += penalty_weight * ca.sumsqr(ca.fmax(0, y - y_max))
    obj += penalty_weight * ca.sumsqr(ca.fmax(0, y_min - y))

    opti.minimize(obj)

    # Unicycle dynamics
    for k in range(N):
        opti.subject_to(x[k+1] == x[k] + dt * v[k]*ca.cos(th[k]))
        opti.subject_to(y[k+1] == y[k] + dt * v[k]*ca.sin(th[k]))
        opti.subject_to(th[k+1] == th[k] + dt * w[k])

    # Boundary conditions
    opti.subject_to(x[0] == x0)
    opti.subject_to(y[0] == y0)
    opti.subject_to(th[0] == th0)
    opti.subject_to(x[N] == xT)
    opti.subject_to(y[N] == yT)
    opti.subject_to(th[N] == thT)

    # Bounds
    opti.subject_to(opti.bounded(v_min, v, v_max))
    opti.subject_to(opti.bounded(-w_max, w, w_max))
    opti.subject_to(opti.bounded(thetabound[0], th, thetabound[1]))

    # Initial guess: mostly straight
    opti.set_initial(x, np.linspace(x0, xT, N+1))
    opti.set_initial(y, np.linspace(y0, yT, N+1))
    opti.set_initial(th, np.linspace(th0, thT, N+1))
    opti.set_initial(v, np.full(N, (v_min+v_max)/2))
    opti.set_initial(w, np.zeros(N))

    # Solver
    opti.solver("ipopt", {"ipopt.print_level": 0, "print_time": 0})

    try:
        sol = opti.solve()
    except RuntimeError:
        # print("WARNING: unicycle_S_shape_solver_sharp_turns failed to find solution")
        #return a None trajectory
        return (
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
        )


    return (
        np.array(sol.value(x)),
        np.array(sol.value(y)),
        np.array(sol.value(th)),
        np.array(sol.value(v)),
        np.array(sol.value(w)),
    )



if __name__ == "__main__":
    # Example usage
    x0, y0, th0 = 85.00,54.00,3.56
    xT, yT, thT = 66.26,51.85,3.56

    start = (x0, y0)
    traj_len = 12

    xbound = (35,102)
    ybound = (35,90)


    raw = unicycle_single_int_trajectory_generator(
                start, xbound, ybound, num_u=traj_len - 1)
    
    print("Generated trajectory:")
    for t in range(len(raw)):
        (p_t, p_tp1, (θ_t, θ_tp1), u_t) = raw[t]
        print(f"Step {t}: Pos {p_t} Ori {θ_t:.2f} -> Pos {p_tp1} Ori {θ_tp1:.2f} | Control (v,w): {u_t}")

    print(len(raw))












