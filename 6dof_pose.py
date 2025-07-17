import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
import logging
import time

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# --- Global Variables (For Mouse Corner Selection) ---
selected_points_global = []
window_name_corner_selection = "Corner Selection - 4 points (TopLeft, TopRight, BottomRight, BottomLeft) 'r'->reset, 'q'->confirm"
zoom_window_size = 150  # Zoom window size
zoom_factor = 5  # Zoom factor
tracking_history = []  # To track mouse movements
stability_threshold = 1.5  # Stability threshold

def analyze_point_stability(x, y, tracking_history):
    """Analyzes mouse movement stability"""
    tracking_history.append((x, y))
    if len(tracking_history) > 20:
        tracking_history.pop(0)
    
    if len(tracking_history) < 15:
        return False
    
    diffs = []
    for i in range(1, len(tracking_history)):
        prev_x, prev_y = tracking_history[i-1]
        curr_x, curr_y = tracking_history[i]
        diff = np.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
        diffs.append(diff)
    
    avg_movement = np.mean(diffs)
    return avg_movement < stability_threshold

def mouse_callback_corner_selection(event, x, y, flags, param):
    """Enhanced mouse callback function"""
    global selected_points_global, tracking_history
    
    if event == cv2.EVENT_MOUSEMOVE or event == cv2.EVENT_LBUTTONDOWN:
        display_copy = param['original_image_for_callback'].copy()
        for i, point in enumerate(selected_points_global):
            cv2.circle(display_copy, (int(point[0]), int(point[1])), 7, (0, 255, 0), -1)
            cv2.putText(display_copy, str(i+1), (int(point[0])+10, int(point[1])-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50,50,255), 2)
        
        h, w = param['original_image_for_callback'].shape[:2]
        half_size = zoom_window_size // (2 * zoom_factor)
        x_start = max(0, x - half_size)
        y_start = max(0, y - half_size)
        x_end = min(w, x + half_size)
        y_end = min(h, y + half_size)
        
        if x_start < x_end and y_start < y_end:
            roi = param['original_image_for_callback'][y_start:y_end, x_start:x_end].copy()
            if roi.size > 0:
                zoomed_roi = cv2.resize(roi, (zoom_window_size, zoom_window_size),
                                      interpolation=cv2.INTER_LINEAR)
                cursor_x = int((x - x_start) * zoom_window_size / (x_end - x_start))
                cursor_y = int((y - y_start) * zoom_window_size / (y_end - y_start))
                cv2.line(zoomed_roi, (cursor_x, 0), (cursor_x, zoom_window_size), (0, 0, 255), 1)
                cv2.line(zoomed_roi, (0, cursor_y), (zoom_window_size, cursor_y), (0, 0, 255), 1)
                cv2.circle(zoomed_roi, (cursor_x, cursor_y), 3, (0, 255, 255), 1)
                
                if event == cv2.EVENT_MOUSEMOVE:
                    is_stable = analyze_point_stability(x, y, tracking_history)
                else: 
                    tracking_history.clear()
                    is_stable = False
                
                stability_color = (0, 255, 0) if is_stable else (0, 0, 255)
                cv2.putText(zoomed_roi, f"({x}, {y})", (5, 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.putText(zoomed_roi, "STABLE" if is_stable else "UNSTABLE",
                           (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, stability_color, 1)
                
                zoom_roi_x = w - zoom_window_size - 10
                zoom_roi_y = 10
                display_copy[zoom_roi_y:zoom_roi_y+zoom_window_size, 
                           zoom_roi_x:zoom_roi_x+zoom_window_size] = zoomed_roi
                cv2.rectangle(display_copy, (x_start, y_start), (x_end, y_end), (0,255,255), 1)
        
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(selected_points_global) < 4:
                selected_points_global.append((float(x), float(y)))
                if len(selected_points_global) == 4:
                    cv2.putText(display_copy, "4 points selected. Press 'q' to confirm, 'r' to reset",
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
            else:
                logging.warning("4 points already selected. Press 'q' to confirm, 'r' to reset.")
        
        param['last_drawn_image'] = display_copy.copy()
        cv2.imshow(window_name_corner_selection, display_copy)

def get_initial_2d_corners_interactive(image_to_select_on):
    global selected_points_global
    selected_points_global = []
    callback_params = {
        'original_image_for_callback': image_to_select_on.copy(),
        'last_drawn_image': image_to_select_on.copy()
    }
    cv2.namedWindow(window_name_corner_selection)
    cv2.setMouseCallback(window_name_corner_selection, mouse_callback_corner_selection, callback_params)
    logging.info("Please select on the image the 4 corners of the plate in the following order:\n"
                 "  1. Click: Plate's PHYSICAL Top-Left Corner\n"
                 "  2. Click: Plate's PHYSICAL Top-Right Corner\n"
                 "  3. Click: Plate's PHYSICAL Bottom-Right Corner\n"
                 "  4. Click: Plate's PHYSICAL Bottom-Left Corner\n"
                 "(This order must correspond to the 3D model corner definition!)\n"
                 "After selection, press 'q' to confirm or 'r' to reset while the window is active.")
    current_display_image = image_to_select_on.copy()
    while True:
        cv2.imshow(window_name_corner_selection, current_display_image)
        key = cv2.waitKey(20) & 0xFF
        if 'last_drawn_image' in callback_params:
            current_display_image = callback_params['last_drawn_image']
        if key == ord('q'):
            if len(selected_points_global) == 4: break
            else: 
                logging.warning(f"Please select {4 - len(selected_points_global)} more point(s).")
        elif key == ord('r'):
            selected_points_global = []
            current_display_image = image_to_select_on.copy()
            callback_params['last_drawn_image'] = current_display_image
            logging.info("Selection reset. Please re-select the 4 corners.")
    cv2.destroyWindow(window_name_corner_selection)
    return np.array(selected_points_global, dtype=np.float32)

def load_camera_parameters(matrix_path, dist_coeffs_path):
    try:
        camera_matrix = np.load(matrix_path, allow_pickle=True)
        dist_coeffs = np.load(dist_coeffs_path, allow_pickle=True)
        logging.info(f"Camera parameters loaded from '{matrix_path}' and '{dist_coeffs_path}'.")
        return camera_matrix, dist_coeffs
    except Exception as e:
        logging.error(f"Error loading camera parameters: {e}", exc_info=True)
        raise

def define_object_3d_points(width, height):
    half_w, half_h = width / 2.0, height / 2.0
    object_points = np.array([
        [-half_w,  half_h, 0.0], [-half_w,  half_h, 0.001], 
        [ half_w,  half_h, 0.0], [ half_w,  half_h, 0.001], 
        [ half_w, -half_h, 0.0], [ half_w, -half_h, 0.001], 
        [-half_w, -half_h, 0.0], [-half_w, -half_h, 0.001]  
    ], dtype=np.float32)
    return object_points

def rodrigues_to_euler(rvec, convention='ZYX'):
    try:
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        r = R.from_matrix(rotation_matrix)
        euler_angles = r.as_euler(convention, degrees=True)
        return euler_angles, rotation_matrix
    except Exception as e:
        logging.error(f"Error converting from Rodrigues to Euler: {e}, rvec: {rvec}", exc_info=True)
        return np.array([np.nan, np.nan, np.nan]), np.eye(3)

def point_reprojection_error_func(params, object_points_3d, image_points_2d, camera_matrix, dist_coeffs_optim,
                               point_weights=None, fixed_distortion_coeffs=None):
    rvec = params[:3].reshape(3, 1)
    tvec = params[3:6].reshape(3, 1)
    current_dist_coeffs_for_cv = fixed_distortion_coeffs
    if dist_coeffs_optim is not None:
        if isinstance(dist_coeffs_optim, np.ndarray) and dist_coeffs_optim.size > 0:
            current_dist_coeffs_for_cv = dist_coeffs_optim
        else:
            logging.warning("dist_coeffs_optim in point_reprojection_error_func was not None or a valid array. Relying on fixed_distortion_coeffs if available.")
    if current_dist_coeffs_for_cv is None:
        current_dist_coeffs_for_cv = np.zeros((5,1), dtype=np.float32) 
    projected_points, _ = cv2.projectPoints(object_points_3d, rvec, tvec, camera_matrix, current_dist_coeffs_for_cv)
    projected_points = projected_points.reshape(-1, 2)
    error = (projected_points - image_points_2d).flatten()
    if point_weights is not None:
        if point_weights.size == error.size:
            error = error * point_weights
        elif point_weights.size == error.size // 2:
            error = error * np.repeat(point_weights, 2)
    return error

def analytical_jacobian_points(params, object_points_3d, image_points_2d, camera_matrix,
                               dist_coeffs_optim=None, point_weights=None, fixed_distortion_coeffs=None):
    rvec = params[:3].reshape(3, 1)
    tvec = params[3:6].reshape(3, 1)
    dist = dist_coeffs_optim if dist_coeffs_optim is not None else (fixed_distortion_coeffs if fixed_distortion_coeffs is not None else np.zeros((5,1)))
    _, J_full = cv2.projectPoints(object_points_3d, rvec, tvec, camera_matrix, dist, jacobian=True)
    J_rt = J_full[:, :6].copy()
    if point_weights is not None:
        w = np.array(point_weights).flatten()
        if w.size == object_points_3d.shape[0]: w = np.repeat(w, 2)
        J_rt = J_rt * w[:, np.newaxis]
    return J_rt

def line_reprojection_error(params, object_3d_edges, detected_2d_lines, camera_matrix,
                            dist_coeffs, line_weights=None):
    rvec = params[:3].reshape(3, 1)
    tvec = params[3:6].reshape(3, 1)
    current_dist_coeffs_for_cv = dist_coeffs
    if current_dist_coeffs_for_cv is None:
        current_dist_coeffs_for_cv = np.zeros((5,1), dtype=np.float32)
    edges = np.asarray(object_3d_edges, dtype=np.float32)
    M = edges.shape[0]
    pts3d = edges.reshape(-1, 3)
    pts2d_proj, _ = cv2.projectPoints(pts3d, rvec, tvec, camera_matrix, current_dist_coeffs_for_cv)
    pts2d_proj = pts2d_proj.reshape(-1, 2)
    residuals = []
    lines = np.asarray(detected_2d_lines, dtype=np.float32).reshape(-1, 4)
    for i in range(M):
        x1, y1, x2, y2 = lines[i]
        dx, dy = x2 - x1, y2 - y1
        norm = np.hypot(dx, dy)
        if norm == 0:
            residuals.extend([0,0])
            continue
        p1_proj = pts2d_proj[2*i]
        p2_proj = pts2d_proj[2*i + 1]
        d1 = abs(dx * (p1_proj[1] - y1) - dy * (p1_proj[0] - x1)) / norm
        d2 = abs(dx * (p2_proj[1] - y1) - dy * (p2_proj[0] - x1)) / norm
        residuals.extend([d1, d2])
    residuals = np.array(residuals, dtype=np.float32)
    if line_weights is not None:
        w = np.array(line_weights).flatten()
        if w.size == M: w = np.repeat(w, 2)
        residuals = residuals * w
    return residuals

def combined_objective_function(params, object_points_3d, image_points_2d, camera_matrix,
                                fixed_distortion_coeffs, use_analytical_jacobian=False, 
                                point_weights=None, object_3d_edges_indices=None,
                                all_3d_points_for_edges=None, detected_2d_lines=None,
                                line_term_weight=0.1, line_weights=None,
                                num_base_params=6, optimize_distortion=False):
    dist_coeffs_to_pass_as_optim_arg_for_point_func = None
    fixed_coeffs_to_pass_as_fixed_arg_for_point_func = None
    current_distortion_coeffs_for_projection = None

    if optimize_distortion:
        num_dist_coeffs_in_params = len(params) - num_base_params
        if num_dist_coeffs_in_params > 0:
            current_distortion_coeffs_for_projection = params[num_base_params:].reshape(-1, 1)
        else:
            logging.error("CRITICAL: optimize_distortion is True, but no distortion coefficients in 'params'. Using zeros.")
            current_distortion_coeffs_for_projection = np.zeros((5, 1), dtype=params.dtype) 
        dist_coeffs_to_pass_as_optim_arg_for_point_func = current_distortion_coeffs_for_projection
        fixed_coeffs_to_pass_as_fixed_arg_for_point_func = None
    else:
        if fixed_distortion_coeffs is None:
            logging.warning("optimize_distortion is False, but 'fixed_distortion_coeffs' is None. Using zeros.")
            current_distortion_coeffs_for_projection = np.zeros((5, 1), dtype=np.float32)
        else:
            current_distortion_coeffs_for_projection = fixed_distortion_coeffs
        dist_coeffs_to_pass_as_optim_arg_for_point_func = None
        fixed_coeffs_to_pass_as_fixed_arg_for_point_func = current_distortion_coeffs_for_projection

    pt_params_for_point_func = params[:num_base_params]
    error_points = point_reprojection_error_func(
        params=pt_params_for_point_func, 
        object_points_3d=object_points_3d,
        image_points_2d=image_points_2d,
        camera_matrix=camera_matrix,
        dist_coeffs_optim=dist_coeffs_to_pass_as_optim_arg_for_point_func,
        point_weights=point_weights,
        fixed_distortion_coeffs=fixed_coeffs_to_pass_as_fixed_arg_for_point_func
    )
    residuals = [error_points]
    if object_3d_edges_indices and detected_2d_lines is not None and all_3d_points_for_edges is not None:
        edges_3d = []
        for (i1, i2) in object_3d_edges_indices:
            p1 = all_3d_points_for_edges[i1]
            p2 = all_3d_points_for_edges[i2]
            edges_3d.append((p1, p2))
        error_lines = line_reprojection_error(
            params=pt_params_for_point_func, 
            object_3d_edges=edges_3d,
            detected_2d_lines=detected_2d_lines,
            camera_matrix=camera_matrix,
            dist_coeffs=current_distortion_coeffs_for_projection, 
            line_weights=line_weights
        )
        residuals.append(line_term_weight * error_lines)
    return np.concatenate(residuals, axis=0)

def run_optimization(initial_params, object_points_3d_inliers, image_points_2d_inliers,
                     camera_matrix, dist_coeffs_for_objective, point_weights=None,
                     f_scale_loss=1.0, max_irls_iterations=5, convergence_tol=1e-3,
                     optimize_distortion_flag_local=False): 
    M = object_points_3d_inliers.shape[0]
    if M == 0 : # No points to optimize
        logging.warning("run_optimization called with zero inlier points. Skipping.")
        # Return a structure similar to opt_result but indicating failure
        class DummyOptResult:
            def __init__(self):
                self.success = False
                self.x = initial_params
                self.cost = float('inf')
                self.optimality = float('inf')
                self.jac = None
                self.message = "No inlier points for optimization."
        return DummyOptResult()

    if point_weights is None:
        current_weights = np.ones(M, dtype=np.float64)
    else:
        current_weights = point_weights.astype(np.float64).copy()
    
    params_iter = initial_params.copy() 
    opt_result = None
    num_base_params = 6 
    fixed_coeffs_to_pass_to_combined_func = None
    if optimize_distortion_flag_local:
        fixed_coeffs_to_pass_to_combined_func = None
    else:
        fixed_coeffs_to_pass_to_combined_func = dist_coeffs_for_objective

    for iter_num in range(max_irls_iterations):
        residual_weights = np.repeat(current_weights, 2)
        current_args_for_least_squares = (
            object_points_3d_inliers, image_points_2d_inliers, camera_matrix,
            fixed_coeffs_to_pass_to_combined_func, False, residual_weights,
            None, None, None, 0.0, None, num_base_params, optimize_distortion_flag_local
        )
        opt_result = least_squares(
            combined_objective_function, params_iter, jac='2-point', method='lm',
            loss='linear', args=current_args_for_least_squares, verbose=0
        )
        if not opt_result.success:
            logging.warning(f"  IRLS iter {iter_num+1}: least_squares failed. Reason: {opt_result.message}")
            break 
        res = opt_result.fun.reshape(-1, 2) 
        norms = np.linalg.norm(res, axis=1) 
        new_weights = np.ones_like(current_weights) 
        threshold = f_scale_loss 
        large_error_indices = norms > threshold
        if np.any(large_error_indices): 
            safe_norms = norms[large_error_indices]
            safe_norms[safe_norms == 0] = 1e-9 
            new_weights[large_error_indices] = threshold / safe_norms
        if np.linalg.norm(new_weights - current_weights) < convergence_tol:
            logging.info(f"  IRLS converged in iteration {iter_num+1}.")
            params_iter = opt_result.x
            break
        current_weights = new_weights
        params_iter = opt_result.x
        if iter_num == max_irls_iterations - 1:
            logging.info("  IRLS reached maximum iterations.")
    return opt_result

def main_pose_estimation_pipeline(image_path, cam_matrix_path, cam_dist_path, object_width, object_height):
    logging.info("Pipeline starting...")
    start_time_pipeline = time.perf_counter() ## <-- DEĞİŞİKLİK: time.perf_counter() KULLANILDI

    
    # Added code to print original image size
    original_image = cv2.imread(image_path)
    if original_image is not None:
        height, width = original_image.shape[:2]
        logging.info(f"Original image size: {width}x{height} pixels")

    camera_matrix, dist_coeffs_calib = load_camera_parameters(cam_matrix_path, cam_dist_path)
    # Ensure dist_coeffs_calib is a valid array, default to 5 zeros if None or empty
    if dist_coeffs_calib is None or dist_coeffs_calib.size == 0:
        dist_coeffs_calib = np.zeros((5, 1), dtype=np.float32)
        logging.info("Invalid calibration distortion coefficients, using zero (5,1) matrix.")
    elif dist_coeffs_calib.ndim == 1: # Ensure it's a column vector
        dist_coeffs_calib = dist_coeffs_calib.reshape(-1,1)


    logging.info(f"Camera Matrix:\n{camera_matrix}")
    logging.info(f"Distortion Coefficients (from calibration):\n{dist_coeffs_calib.flatten()}")

    undistorted_image_for_pnp = cv2.undistort(original_image, camera_matrix, dist_coeffs_calib)
    h_img, w_img = undistorted_image_for_pnp.shape[:2]
    logging.info(f"Image '{image_path}' loaded and undistorted (initially) (size: {w_img}x{h_img}).")

    object_points_3d_model = define_object_3d_points(object_width, object_height)
    pnp_object_points_3d = object_points_3d_model[::2] 
    logging.info(f"3D Object Corners (for PnP):\n{pnp_object_points_3d}")

    raw_image_points_2d = get_initial_2d_corners_interactive(undistorted_image_for_pnp.copy())
    if raw_image_points_2d is None or len(raw_image_points_2d) != 4:
        logging.error("4 corner points were not selected. Terminating program.")
        return None, 0, {}, None
    
    gray_undistorted_image = cv2.cvtColor(undistorted_image_for_pnp, cv2.COLOR_BGR2GRAY)
    criteria_subpix = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
    refined_image_points_2d = cv2.cornerSubPix(gray_undistorted_image, raw_image_points_2d.reshape(-1,1,2), (5,5), (-1,-1), criteria_subpix)
    refined_image_points_2d = refined_image_points_2d.reshape(-1,2)

    # ... (Corner matching visualization unchanged) ...
    vis_image_corner_matching = undistorted_image_for_pnp.copy()
    corner_labels_3d = ["3D_SU", "3D_SAU", "3D_SAA", "3D_SA"] 
    for i, p_refined in enumerate(refined_image_points_2d):
        text_to_show = f"Selection {i+1} -> {corner_labels_3d[i]}"
        cv2.circle(vis_image_corner_matching, (int(p_refined[0]), int(p_refined[1])), 7, (0, 255, 0), -1)
        cv2.putText(vis_image_corner_matching, text_to_show,
                    (int(p_refined[0]) - 60, int(p_refined[1]) - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
    cv2.imshow("Refined Corners and 3D Model Correspondence", vis_image_corner_matching)
    logging.info("Displaying refined corners and 3D model correspondences.")
    logging.info("Please verify that the selected points (Selection 1,2,3,4) match the correct 3D corner labels (3D_SU, 3D_SAU, etc.)!")
    logging.info("Press any key to continue...")
    cv2.waitKey(0)
    cv2.destroyWindow("Refined Corners and 3D Model Correspondence")

    pnp_object_points_3d_float = pnp_object_points_3d.astype(np.float32)
    refined_image_points_2d_float = refined_image_points_2d.astype(np.float32)
    pnp_seeds = []
    pnp_dist_coeffs_for_solvepnp = None 
    pnp_algorithms = {
        "ITERATIVE": {"flag": cv2.SOLVEPNP_ITERATIVE, "min_points": 4}, "EPNP": {"flag": cv2.SOLVEPNP_EPNP, "min_points": 4},
        "AP3P": {"flag": cv2.SOLVEPNP_AP3P, "min_points": 3}, "IPPE": {"flag": cv2.SOLVEPNP_IPPE, "min_points": 4},
        "IPPE_SQUARE": {"flag": cv2.SOLVEPNP_IPPE_SQUARE, "min_points": 4}, 
    }
    obj_pts_for_pnp_generic = pnp_object_points_3d_float.reshape(-1, 3)
    img_pts_for_pnp_generic = refined_image_points_2d_float.reshape(-1, 2)
    for name, algo_info in pnp_algorithms.items():
        if len(refined_image_points_2d_float) >= algo_info["min_points"]:
            try:
                start_time_pnp = time.perf_counter() ## <-- DEĞİŞİKLİK: time.perf_counter() KULLANILDI
                nsol, rvecs, tvecs, _ = cv2.solvePnPGeneric(
                    obj_pts_for_pnp_generic, img_pts_for_pnp_generic, camera_matrix, 
                    pnp_dist_coeffs_for_solvepnp, flags=algo_info["flag"])
                end_time_pnp = time.perf_counter()   ## <-- DEĞİŞİKLİK: time.perf_counter() KULLANILDI
                exec_time_ms = (end_time_pnp - start_time_pnp) * 1000

                if nsol > 0:
                    avg_time_per_solution = exec_time_ms / nsol
                    for i in range(nsol):
                        params_init_eval = np.concatenate((rvecs[i].flatten(), tvecs[i].flatten()))
                        err_vals = point_reprojection_error_func(params_init_eval, obj_pts_for_pnp_generic, img_pts_for_pnp_generic, 
                                                               camera_matrix, None, None, None)
                        mean_abs_err = np.mean(np.abs(err_vals)) if err_vals.size > 0 else np.inf
                        pnp_seeds.append({"name": f"{name}_{i}", "rvec": rvecs[i], "tvec": tvecs[i], 
                                          "inliers_mask": np.ones(len(img_pts_for_pnp_generic), dtype=bool), 
                                          "reprojection_error_initial": mean_abs_err,
                                          "execution_time_ms": avg_time_per_solution})
                        logging.info(f"{name}_{i}: solution found. Error: {mean_abs_err:.4f}, Time: {avg_time_per_solution:.4f} ms")
            except cv2.error as e: logging.warning(f"{name} PnP (Generic) error: {e}")
    ransac_methods = {
        "RANSAC_EPNP": cv2.SOLVEPNP_EPNP, "RANSAC_P3P": cv2.SOLVEPNP_P3P,
        "RANSAC_ITERATIVE": cv2.SOLVEPNP_ITERATIVE, "RANSAC_IPPE": cv2.SOLVEPNP_IPPE,
    }
    obj_pts_for_ransac_all = pnp_object_points_3d_float.reshape(-1,1,3) 
    img_pts_for_ransac_all = refined_image_points_2d_float.reshape(-1,1,2)
    for name, flag in ransac_methods.items():
        min_pts_base = 3 if flag == cv2.SOLVEPNP_P3P else 4
        if len(refined_image_points_2d_float) < min_pts_base: continue
        try:
            start_time_ransac = time.perf_counter() ## <-- DEĞİŞİKLİK: time.perf_counter() KULLANILDI
            succ, rvec, tvec, inliers_idx = cv2.solvePnPRansac(
                obj_pts_for_ransac_all, img_pts_for_ransac_all, camera_matrix, pnp_dist_coeffs_for_solvepnp,
                iterationsCount=200, reprojectionError=5.0, confidence=0.99, flags=flag)
            end_time_ransac = time.perf_counter()   ## <-- DEĞİŞİKLİK: time.perf_counter() KULLANILDI
            exec_time_ms_ransac = (end_time_ransac - start_time_ransac) * 1000

            if succ:
                if inliers_idx is None or len(inliers_idx) < min_pts_base: continue
                obj_in = pnp_object_points_3d_float[inliers_idx.flatten()]
                img_in = refined_image_points_2d_float[inliers_idx.flatten()]
                params_init_ransac = np.concatenate((rvec.flatten(), tvec.flatten()))
                err_vals_ransac = point_reprojection_error_func(params_init_ransac, obj_in, img_in, camera_matrix, None, None, None)
                mean_abs_err_ransac = np.mean(np.abs(err_vals_ransac)) if err_vals_ransac.size > 0 else np.inf
                mask = np.zeros(len(refined_image_points_2d_float), dtype=bool)
                mask[inliers_idx.flatten()] = True
                pnp_seeds.append({"name": name, "rvec": rvec, "tvec": tvec, "inliers_mask": mask, 
                                  "reprojection_error_initial": mean_abs_err_ransac,
                                  "execution_time_ms": exec_time_ms_ransac})
                logging.info(f"{name}: solution found. Inliers: {np.sum(mask)}, Error: {mean_abs_err_ransac:.4f}, Time: {exec_time_ms_ransac:.4f} ms")
        except cv2.error as e: logging.warning(f"{name} RANSAC PnP error: {e}")

    if not pnp_seeds:
        logging.error("No PnP/RANSAC method produced an initial pose.")
        return None, 0, {}, None
    pnp_seeds.sort(key=lambda s: s["reprojection_error_initial"])
    all_optimized_results = []
    
    # Global flag for preferring distortion optimization
    global_optimize_distortion_preference = True 
    num_dist_coeffs_to_optimize = 5 # Standard number of distortion coeffs (k1,k2,p1,p2,k3)

    logging.info(f"\nStarting non-linear optimization for all valid PnP seeds (Distortion optimization preference: {global_optimize_distortion_preference})...")
    
    for seed_idx, seed in enumerate(pnp_seeds):
        logging.info(f"-- Optimization Candidate {seed_idx+1}/{len(pnp_seeds)}: {seed['name']} (Initial Error: {seed['reprojection_error_initial']:.4f}) --")
        current_obj_points_3d_optim = pnp_object_points_3d_float[seed["inliers_mask"]]
        current_img_points_2d_optim = refined_image_points_2d_float[seed["inliers_mask"]]
        
        if len(current_obj_points_3d_optim) < 3:
            logging.warning(f"Too few inliers for seed {seed['name']} ({len(current_obj_points_3d_optim)}), skipping optimization.")
            continue

        initial_rvec_tvec_for_optim = np.concatenate((seed["rvec"].flatten(), seed["tvec"].flatten()))
        initial_dist_coeffs_guess = np.zeros(num_dist_coeffs_to_optimize, dtype=np.float32)
        if dist_coeffs_calib is not None:
            coeffs_flat = dist_coeffs_calib.flatten()
            len_to_copy = min(len(coeffs_flat), num_dist_coeffs_to_optimize)
            initial_dist_coeffs_guess[:len_to_copy] = coeffs_flat[:len_to_copy]

        local_run_optimize_distortion = global_optimize_distortion_preference
        num_pose_dof = 6
        num_variables_for_optim = num_pose_dof
        if local_run_optimize_distortion:
            num_variables_for_optim += num_dist_coeffs_to_optimize
        
        num_residuals_available = 2 * len(current_img_points_2d_optim)

        if local_run_optimize_distortion and num_residuals_available < num_variables_for_optim:
            logging.warning(f"Seed {seed['name']}: Insufficient residuals ({num_residuals_available}) for {num_variables_for_optim} variables (pose+distortion). Distortion optimization has been temporarily disabled for this seed.")
            local_run_optimize_distortion = False

        if local_run_optimize_distortion:
            params_for_run_optimization = np.concatenate((initial_rvec_tvec_for_optim, initial_dist_coeffs_guess))
            dist_coeffs_arg_for_run_opt = initial_dist_coeffs_guess.reshape(-1,1) 
        else: 
            params_for_run_optimization = initial_rvec_tvec_for_optim.copy()
            dist_coeffs_arg_for_run_opt = dist_coeffs_calib
        
        opt_result = run_optimization(params_for_run_optimization, 
                                      current_obj_points_3d_optim, current_img_points_2d_optim,
                                      camera_matrix, dist_coeffs_arg_for_run_opt, 
                                      f_scale_loss=1.5, max_irls_iterations=10, convergence_tol=1e-4,
                                      optimize_distortion_flag_local=local_run_optimize_distortion)

        optimized_rvec_o2c_final = seed["rvec"]
        optimized_tvec_o2c_final = seed["tvec"]
        final_optimized_dist_coeffs = dist_coeffs_calib
        final_cost_metric = float('inf')
        final_optimality_metric = float('inf')
        final_jacobian_matrix = None
        is_optimization_successful = False

        if opt_result and opt_result.success:
            is_optimization_successful = True
            optimized_params_vector = opt_result.x
            optimized_rvec_o2c_final = optimized_params_vector[:3].reshape(3, 1)
            optimized_tvec_o2c_final = optimized_params_vector[3:6].reshape(3, 1)

            if local_run_optimize_distortion:
                final_optimized_dist_coeffs = optimized_params_vector[num_pose_dof : num_pose_dof + num_dist_coeffs_to_optimize].reshape(-1,1)
                logging.info(f"  Optimized Distortion Coefficients: {final_optimized_dist_coeffs.flatten()}")
            final_cost_metric = opt_result.cost if opt_result.cost is not None else float('inf')
            final_optimality_metric = opt_result.optimality if opt_result.optimality is not None else float('inf')
            final_jacobian_matrix = opt_result.jac
            logging.info(f"  OPTIMIZATION SUCCESSFUL: rvec={optimized_rvec_o2c_final.flatten()}, tvec={optimized_tvec_o2c_final.flatten()}")
            logging.info(f"  Final Cost (0.5 * sum(sq_res)): {final_cost_metric:.4e}, Optimality: {final_optimality_metric:.4e}")
        else:
            logging.warning(f"  OPTIMIZATION FAILED. Message: {opt_result.message if opt_result else 'Exception/No result'}")
            if seed["reprojection_error_initial"] != np.inf:
                 num_res_approx_fail = len(current_img_points_2d_optim) * 2
                 final_cost_metric = 0.5 * num_res_approx_fail * (seed["reprojection_error_initial"]**2) if num_res_approx_fail > 0 else float('inf')

        param_std_devs_rt_final = np.full(6, float('nan')) 
        trace_cov_rt_final = float('inf')
        if final_jacobian_matrix is not None and is_optimization_successful:
            try:
                num_res_cov = final_jacobian_matrix.shape[0]
                num_total_params_cov = len(opt_result.x)
                deg_freedom_cov = num_res_cov - num_total_params_cov
                if deg_freedom_cov > 0:
                    mse_cov = 2 * final_cost_metric / deg_freedom_cov 
                    hessian_inv_full_cov = np.linalg.pinv(final_jacobian_matrix.T @ final_jacobian_matrix)
                    cov_matrix_scaled_full_cov = hessian_inv_full_cov * mse_cov
                    param_variances_full_cov = np.diag(cov_matrix_scaled_full_cov).copy()
                    param_variances_full_cov[param_variances_full_cov < 0] = 1e-12 
                    param_std_devs_full_cov = np.sqrt(param_variances_full_cov)
                    param_std_devs_rt_final = param_std_devs_full_cov[:num_pose_dof] 
                    trace_cov_rt_final = np.sum(np.diag(cov_matrix_scaled_full_cov)[:num_pose_dof]) 
                    logging.info(f"  Parameter Std Devs (rvec, tvec): {param_std_devs_rt_final}")
                    logging.info(f"  Trace of Covariance Matrix (rvec, tvec): {trace_cov_rt_final:.4e}")
            except Exception as e: logging.error(f"  Error computing covariance: {e}", exc_info=True)
        
        euler_o2c_ypr_final, R_o2c_final = rodrigues_to_euler(optimized_rvec_o2c_final, convention='ZYX')
        R_c2o_final = R_o2c_final.T
        t_c2o_final = -R_c2o_final @ optimized_tvec_o2c_final
        rvec_c2o_final, _ = cv2.Rodrigues(R_c2o_final)
        euler_c2o_ypr_final, _ = rodrigues_to_euler(rvec_c2o_final, convention='ZYX')

        all_optimized_results.append({
            "source_seed_name": seed["name"], "initial_reprojection_error": seed["reprojection_error_initial"],
            "pnp_execution_time_ms": seed.get("execution_time_ms", -1.0),
            "rvec_o2c": optimized_rvec_o2c_final, "tvec_o2c": optimized_tvec_o2c_final,
            "R_o2c": R_o2c_final, "euler_o2c_ypr": euler_o2c_ypr_final,
            "rvec_c2o": rvec_c2o_final, "tvec_c2o": t_c2o_final,
            "R_c2o": R_c2o_final, "euler_c2o_ypr": euler_c2o_ypr_final,
            "optimized_dist_coeffs": final_optimized_dist_coeffs, "cost": final_cost_metric, 
            "optimality": final_optimality_metric, "param_std_devs_rt": param_std_devs_rt_final, 
            "trace_cov_rt": trace_cov_rt_final, "num_inliers_for_optim": len(current_obj_points_3d_optim),
            "optimization_successful": is_optimization_successful,
            "distortion_optimized_in_run": local_run_optimize_distortion
        })

    if not all_optimized_results:
        logging.error("No candidate could be optimized or produced a valid result.")
        return None, 0, {}, None
    all_optimized_results.sort(key=lambda res: res["cost"])
    alg_summary = {}
    for seed_entry_sum in pnp_seeds: 
        algo_base_name_sum = seed_entry_sum["name"].split('_')[0] 
        if "RANSAC" in algo_base_name_sum: 
            method_part_sum = seed_entry_sum["name"].split('_')
            if len(method_part_sum) > 1 : algo_base_name_sum = f"RANSAC_{method_part_sum[1]}" 
            else: algo_base_name_sum = "RANSAC"
        if algo_base_name_sum not in alg_summary:
            alg_summary[algo_base_name_sum] = {"count": 0, "initial_error_list": []}
        alg_summary[algo_base_name_sum]["count"] += 1
        if seed_entry_sum["reprojection_error_initial"] != np.inf:
            alg_summary[algo_base_name_sum]["initial_error_list"].append(seed_entry_sum["reprojection_error_initial"])
    for algo_sum in alg_summary:
        if alg_summary[algo_sum]["initial_error_list"]:
            alg_summary[algo_sum]["mean_initial_error"] = float(np.mean(alg_summary[algo_sum]["initial_error_list"]))
        else: alg_summary[algo_sum]["mean_initial_error"] = float('nan')

    end_time_pipeline = time.perf_counter() ## <-- DEĞİŞİKLİK: time.perf_counter() KULLANILDI
    pipeline_duration = end_time_pipeline - start_time_pipeline
    logging.info(f"Pipeline completed. Total duration: {pipeline_duration:.2f} seconds.")
    return all_optimized_results, pipeline_duration, alg_summary, undistorted_image_for_pnp

if __name__ == "__main__":
    image_file_path = "img2.png" 
    camera_matrix_path = "cameraMatrix_best.pkl" 
    dist_coeffs_path = "dist1.pkl" 
    "" 
    # plate_actual_width = 0.179 
    # plate_actual_height = 0.101
    plate_actual_width = 0.125
    plate_actual_height = 0.195
    all_results_sorted, total_pipeline_time, pnp_stats_summary, vis_undistorted_img = main_pose_estimation_pipeline(
        image_file_path, camera_matrix_path, dist_coeffs_path, plate_actual_width, plate_actual_height
    )

    if all_results_sorted:
        logging.info(f"\n--- ALL OPTIMIZED RESULTS ({len(all_results_sorted)} entries, sorted by cost) ---")
        best_result_for_vis = all_results_sorted[0]
        logging.info(f"\nDrawing axes for the best result ({best_result_for_vis['source_seed_name']}).")
        _camera_matrix_vis, _ = load_camera_parameters(camera_matrix_path, dist_coeffs_path)
        
        vis_image_axes = vis_undistorted_img.copy()
        axis_length = min(plate_actual_width, plate_actual_height) * 0.7

        # 3D uzayda x, y, z ekseni uç noktaları (objeye göre)
        axes_3d = np.float32([
            [0, 0, 0],                          # orijin
            [axis_length, 0, 0],               # x ekseni
            [0, axis_length, 0],               # y ekseni
            [0, 0, axis_length]                # z ekseni
        ]).reshape(-1, 3)

        # Görüntüdeki noktalara projekte et
        imgpts, _ = cv2.projectPoints(axes_3d,
                                    best_result_for_vis['rvec_o2c'],
                                    best_result_for_vis['tvec_o2c'],
                                    _camera_matrix_vis,
                                    None)

        # Noktaları aç
        origin = tuple(imgpts[0].ravel().astype(int))
        x_axis = tuple(imgpts[1].ravel().astype(int))
        y_axis = tuple(imgpts[2].ravel().astype(int))
        z_axis = tuple(imgpts[3].ravel().astype(int))

        # Ok başlı çizgiler
        cv2.arrowedLine(vis_image_axes, origin, x_axis, (0, 0, 255), 3, tipLength=0.1)  # X - kırmızı
        cv2.arrowedLine(vis_image_axes, origin, y_axis, (0, 255, 0), 3, tipLength=0.1)  # Y - yeşil
        cv2.arrowedLine(vis_image_axes, origin, z_axis, (255, 0, 0), 3, tipLength=0.1)  # Z - mavi
        cv2.imshow(f"Best Result ({best_result_for_vis['source_seed_name']}) - Pose and Axes", vis_image_axes)
        logging.info("Displaying visualization of the best result. Press any key to continue and view further summaries...")
        cv2.waitKey(0)
        cv2.destroyWindow(f"Best Result ({best_result_for_vis['source_seed_name']}) - Pose and Axes")

        for rank, result_item in enumerate(all_results_sorted):
            logging.info(f"\n--- DETAILED RESULT (Rank: {rank+1}/{len(all_results_sorted)}, Source: {result_item['source_seed_name']}) ---")
            left_summary_lines = []
            right_summary_lines = []
            left_summary_lines.append(f"Source Seed: {result_item['source_seed_name']} (Rank: {rank+1})")
            left_summary_lines.append(f"PnP Initial Solve Time: {result_item['pnp_execution_time_ms']:.4f} ms") ## <-- DEĞİŞİKLİK: Daha fazla hassasiyet için .4f
            left_summary_lines.append(f"Optimization Successful: {'Yes' if result_item['optimization_successful'] else 'No'}")
            left_summary_lines.append(f"Distortion Optimized (in this run): {'Yes' if result_item['distortion_optimized_in_run'] else 'No'}")
            left_summary_lines.append(f"Initial Reprojection Error: {result_item['initial_reprojection_error']:.4f} pixels")
            left_summary_lines.append(f"Final Cost: {result_item['cost']:.4e}")
            left_summary_lines.append(f"Optimality: {result_item['optimality']:.4e}")
            left_summary_lines.append(f"Number of Inliers Used: {result_item['num_inliers_for_optim']}")
            left_summary_lines.append(f"Std Dev (Pose): {np.array2string(result_item['param_std_devs_rt'], precision=3, suppress_small=True)}")
            if result_item.get('optimized_dist_coeffs') is not None:
                 dist_coeffs_str = np.array2string(result_item['optimized_dist_coeffs'].flatten(), precision=4, suppress_small=True)
                 left_summary_lines.append(f"Used/Optimized Distortion Coefficients: {dist_coeffs_str}")
            left_summary_lines.append("-" * 30)
            left_summary_lines.append("Object to Camera (O2C):") 
            left_summary_lines.append(f"  rvec: {np.array2string(result_item['rvec_o2c'].flatten(), precision=3, suppress_small=True)}")
            left_summary_lines.append(f"  tvec: {np.array2string(result_item['tvec_o2c'].flatten(), precision=3, suppress_small=True)} (m)")
            euler_o2c = result_item['euler_o2c_ypr']
            left_summary_lines.append(f"  Euler (ZYX): Yaw:{euler_o2c[0]:.2f}, Pitch:{euler_o2c[1]:.2f}, Roll:{euler_o2c[2]:.2f} (deg)")
            left_summary_lines.append("")
            left_summary_lines.append("Camera to Object (C2O):")
            left_summary_lines.append(f"  rvec: {np.array2string(result_item['rvec_c2o'].flatten(), precision=3, suppress_small=True)}")
            left_summary_lines.append(f"  tvec: {np.array2string(result_item['tvec_c2o'].flatten(), precision=3, suppress_small=True)} (m)")
            euler_c2o = result_item['euler_c2o_ypr']
            left_summary_lines.append(f"  Euler (ZYX): Yaw:{euler_c2o[0]:.2f}, Pitch:{euler_c2o[1]:.2f}, Roll:{euler_c2o[2]:.2f} (deg)")
            left_summary_lines.append("-" * 30)
            R_ref_to_cam = np.eye(3)
            t_ref_to_cam = np.array([1.0, 0.0, 0.0]).reshape(3,1)
            R_ref_to_obj = R_ref_to_cam @ result_item['R_c2o'] 
            t_ref_to_obj = R_ref_to_cam @ result_item['tvec_c2o'] + t_ref_to_cam 
            rvec_ref_to_obj, _ = cv2.Rodrigues(R_ref_to_obj)
            euler_ref_to_obj, _ = rodrigues_to_euler(rvec_ref_to_obj, convention='ZYX')
            left_summary_lines.append("Reference to Object (Ref_to_Obj):")
            left_summary_lines.append("  (Ref: Same rotation as camera, at +1m along the camera's X-axis)")
            left_summary_lines.append(f"  rvec: {np.array2string(rvec_ref_to_obj.flatten(), precision=3, suppress_small=True)}")
            left_summary_lines.append(f"  tvec: {np.array2string(t_ref_to_obj.flatten(), precision=3, suppress_small=True)} (m)")
            left_summary_lines.append(f"  Euler (ZYX): Yaw:{euler_ref_to_obj[0]:.2f}, Pitch:{euler_ref_to_obj[1]:.2f}, Roll:{euler_ref_to_obj[2]:.2f} (deg)")
            right_summary_lines.append("Rotation Matrix (Object to Camera):")
            for row in result_item['R_o2c']: right_summary_lines.append(f"  {np.array2string(row, precision=3, suppress_small=True)}")
            right_summary_lines.append("")
            right_summary_lines.append("Rotation Matrix (Camera to Object):")
            for row in result_item['R_c2o']: right_summary_lines.append(f"  {np.array2string(row, precision=3, suppress_small=True)}")
            right_summary_lines.append("")
            right_summary_lines.append("Rotation Matrix (Reference to Object):")
            for row in R_ref_to_obj: right_summary_lines.append(f"  {np.array2string(row, precision=3, suppress_small=True)}")
            right_summary_lines.append("")
            if rank == 0: 
                right_summary_lines.append("-" * 30)
                right_summary_lines.append("PnP Algorithm Initial Statistics:")
                for algo_name, stats in pnp_stats_summary.items():
                    right_summary_lines.append(f"  {algo_name}: Solution Count = {stats['count']}, Avg. Initial Error = {stats['mean_initial_error']:.4f}")
                right_summary_lines.append("")
                right_summary_lines.append(f"Total Pipeline Duration: {total_pipeline_time:.2f} sec")
            max_lines = max(len(left_summary_lines), len(right_summary_lines))
            win_height = (max_lines + 2) * 25 
            win_width = 1350
            summary_img = np.zeros((win_height, win_width, 3), dtype=np.uint8)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.50
            font_color = (0, 255, 0)
            line_type = 1
            line_height_px = 20
            for idx, line in enumerate(left_summary_lines):
                cv2.putText(summary_img, line, (10, (idx + 1) * line_height_px), font, font_scale, font_color, line_type, cv2.LINE_AA)
            for idx, line in enumerate(right_summary_lines):
                cv2.putText(summary_img, line, (650, (idx + 1) * line_height_px), font, font_scale, font_color, line_type, cv2.LINE_AA)
            summary_window_title = f"Summary (Rank {rank+1}) - {result_item['source_seed_name']}"
            cv2.imshow(summary_window_title, summary_img)
            key = cv2.waitKey(0)
            cv2.destroyWindow(summary_window_title)
            if key == ord('q') or key == 27: 
                logging.info("Exiting as per user request.")
                break
        cv2.destroyAllWindows() 
        logging.info("All result summaries displayed.")
        
    else:
        logging.error("Pipeline did not produce any results.")