#!/usr/bin/env python3
"""
Digitize curves from Effio et al. (2016) Fig. 3a and 3b.

Fig 3a: Both curves are BLUE (dashed=measured, solid=simulated)
Fig 3b: Both curves are BLACK (dashed=measured, solid=simulated)
"""

import numpy as np
from PIL import Image
import fitz  # PyMuPDF
import os

PDF_PATH = os.path.expanduser("~/Downloads/1-s2.0-S0021967315017562-main.pdf")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "radialDG-results-membrane")


def extract_page_image(pdf_path, page_num, dpi=600):
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()
    return img


def crop_figure(img, bbox_frac):
    w, h = img.size
    return img.crop((int(bbox_frac[0] * w), int(bbox_frac[1] * h),
                     int(bbox_frac[2] * w), int(bbox_frac[3] * h)))


def trace_curve_in_bbox(img_array, plot_bbox, color_func,
                         x_range, y_range, margin=5):
    """Trace curves within the plot area, with margin to exclude axes.

    Returns per-column median row positions for all matching pixels.
    When two vertically separated clusters exist, returns both.
    """
    top, bottom, left, right = plot_bbox
    plot_h = bottom - top
    plot_w = right - left

    all_points = []    # (x_data, y_data) for all detected curve pixels
    upper_curve = []   # when two curves are visible
    lower_curve = []

    for col in range(left + margin, right - margin):
        matching_rows = []
        for row in range(top + margin, bottom - margin):
            r, g, b = img_array[row, col, :3]
            if color_func(r, g, b):
                matching_rows.append(row)

        if not matching_rows:
            continue

        # Cluster the matching rows
        clusters = []
        current = [matching_rows[0]]
        for i in range(1, len(matching_rows)):
            if matching_rows[i] - matching_rows[i - 1] <= 3:
                current.append(matching_rows[i])
            else:
                if len(current) >= 2:  # filter out single-pixel noise
                    clusters.append(current)
                current = [matching_rows[i]]
        if len(current) >= 2:
            clusters.append(current)

        if not clusters:
            continue

        x_frac = (col - left) / plot_w
        x_val = x_range[0] + x_frac * (x_range[1] - x_range[0])

        if len(clusters) >= 2:
            # Sort by vertical position
            clusters.sort(key=lambda c: np.median(c))
            c1_med = np.median(clusters[0])
            c2_med = np.median(clusters[-1])

            # Only accept if they're sufficiently separated
            if abs(c2_med - c1_med) > 5:
                y_frac1 = 1.0 - (c1_med - top) / plot_h
                y_frac2 = 1.0 - (c2_med - top) / plot_h
                y_val1 = y_range[0] + y_frac1 * (y_range[1] - y_range[0])
                y_val2 = y_range[0] + y_frac2 * (y_range[1] - y_range[0])
                upper_curve.append((x_val, y_val1))  # higher y_data
                lower_curve.append((x_val, y_val2))  # lower y_data
                continue

        # Single cluster or overlapping: take the median
        med_row = np.median(clusters[0])
        y_frac = 1.0 - (med_row - top) / plot_h
        y_val = y_range[0] + y_frac * (y_range[1] - y_range[0])
        all_points.append((x_val, y_val))

    return all_points, upper_curve, lower_curve


def smooth_curve(points, window=7, n_out=150):
    """Smooth and subsample a list of (x, y) points."""
    if len(points) < window * 3:
        if len(points) < 2:
            return np.array([]), np.array([])
        pts = np.array(points)
        return pts[:, 0], pts[:, 1]

    pts = np.array(points)
    x, y = pts[:, 0], pts[:, 1]

    # Sort by x
    idx = np.argsort(x)
    x, y = x[idx], y[idx]

    # Moving average smoothing
    kernel = np.ones(window) / window
    y_smooth = np.convolve(y, kernel, mode='valid')
    x_smooth = x[window // 2: window // 2 + len(y_smooth)]

    # Subsample
    x_out = np.linspace(x_smooth[0], x_smooth[-1], n_out)
    y_out = np.interp(x_out, x_smooth, y_smooth)

    return x_out, y_out


def is_blue(r, g, b):
    return b > 100 and b > r + 25 and b > g + 25


def is_dark(r, g, b):
    return r < 70 and g < 70 and b < 70


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print("Extracting page 5 from PDF at 600 dpi...")
    full_page = extract_page_image(PDF_PATH, page_num=4, dpi=600)
    w_page, h_page = full_page.size
    print(f"  Page size: {w_page}x{h_page}")

    # ================================================================
    # Fig 3a: Dextran T2000 (blue curves)
    # ================================================================
    print("\n--- Fig 3a: Dextran T2000 ---")
    fig3a = crop_figure(full_page, (0.53, 0.03, 0.97, 0.33))
    fig3a.save(os.path.join(OUTPUT_DIR, "effio_fig3a_crop.png"))
    arr3a = np.array(fig3a)
    h3a, w3a = arr3a.shape[:2]
    print(f"  Crop: {w3a}x{h3a}")

    # Manually calibrated plot bbox from image inspection
    # (top, bottom, left, right) in pixel coordinates of the crop
    bbox_3a = (110, 1700, 230, 1830)
    x_range_a = (0.0, 3.5)
    y_range_a = (0.0, 35.0)

    all_a, upper_a, lower_a = trace_curve_in_bbox(
        arr3a, bbox_3a, is_blue, x_range_a, y_range_a, margin=8)
    print(f"  Single-cluster points: {len(all_a)}")
    print(f"  Two-cluster: upper={len(upper_a)}, lower={len(lower_a)}")

    # The upper curve (higher y_data) is whichever has the higher peak
    # In regions where curves overlap, all_points captures the merged signal
    # Combine: upper_curve points have HIGHER concentration → that's one curve
    # lower_curve points have LOWER concentration → that's the other

    # Build the two curves: merge single-cluster with the appropriate branch
    # When curves are separated: upper=higher y, lower=lower y
    # The simulated peak is slightly earlier and taller → it's typically "upper" on the left,
    # then "lower" on the right after the crossover
    # For now, just output both branches + the merged single-cluster

    x_all_a, y_all_a = smooth_curve(all_a, window=9, n_out=200)
    x_up_a, y_up_a = smooth_curve(upper_a, window=5, n_out=80)
    x_lo_a, y_lo_a = smooth_curve(lower_a, window=5, n_out=80)

    # ================================================================
    # Fig 3b: NaCl (black curves)
    # ================================================================
    print("\n--- Fig 3b: NaCl ---")
    fig3b = crop_figure(full_page, (0.53, 0.375, 0.97, 0.68))
    fig3b.save(os.path.join(OUTPUT_DIR, "effio_fig3b_crop.png"))
    arr3b = np.array(fig3b)
    h3b, w3b = arr3b.shape[:2]
    print(f"  Crop: {w3b}x{h3b}")

    # Find axes precisely
    gray3b = np.mean(arr3b, axis=2)
    # Y-axis: leftmost column with many dark pixels
    yaxis_col = None
    for col in range(w3b // 3):
        if np.sum(gray3b[:, col] < 60) > h3b * 0.4:
            yaxis_col = col
            break
    # X-axis: bottommost row with many dark pixels
    xaxis_row = None
    for row in range(h3b - 1, h3b // 3, -1):
        if np.sum(gray3b[row, :] < 60) > w3b * 0.25:
            xaxis_row = row
            break
    # Right edge: rightmost tick
    right_col = None
    for col in range(w3b - 1, w3b // 2, -1):
        dark_near_axis = np.sum(gray3b[xaxis_row - 10:xaxis_row + 10, col] < 60) if xaxis_row else 0
        if dark_near_axis > 2:
            right_col = col
            break

    print(f"  Y-axis col: {yaxis_col}, X-axis row: {xaxis_row}, Right col: {right_col}")

    # Top of plot: find where the first y-tick is at 6 mS/cm
    # Y ticks are at 0,1,2,3,4,5,6 → 7 ticks, 6 intervals
    # Ticks are short horizontal lines extending from the y-axis
    tick_rows = []
    if yaxis_col and xaxis_row:
        for row in range(50, xaxis_row):
            # Check for tick mark: dark pixel at y-axis column extending a few pixels right
            tick_region = gray3b[row, yaxis_col:yaxis_col + 15]
            if np.sum(tick_region < 60) > 8:
                # Check it's not just the axis line itself
                tick_rows.append(row)

    # Group consecutive tick rows
    if tick_rows:
        tick_positions = []
        current_group = [tick_rows[0]]
        for i in range(1, len(tick_rows)):
            if tick_rows[i] - tick_rows[i - 1] <= 5:
                current_group.append(tick_rows[i])
            else:
                tick_positions.append(int(np.median(current_group)))
                current_group = [tick_rows[i]]
        tick_positions.append(int(np.median(current_group)))
        print(f"  Y-axis tick positions (rows): {tick_positions}")
        top_row = tick_positions[0] if tick_positions else 100
    else:
        top_row = 100

    if yaxis_col and xaxis_row and right_col:
        bbox_3b = (top_row, xaxis_row, yaxis_col, right_col)
    else:
        bbox_3b = (100, 1850, 200, 1970)

    print(f"  Plot bbox: {bbox_3b}")

    x_range_b = (0.0, 6.0)
    y_range_b = (0.0, 6.0)

    all_b, upper_b, lower_b = trace_curve_in_bbox(
        arr3b, bbox_3b, is_dark, x_range_b, y_range_b, margin=10)
    print(f"  Single-cluster points: {len(all_b)}")
    print(f"  Two-cluster: upper={len(upper_b)}, lower={len(lower_b)}")

    x_all_b, y_all_b = smooth_curve(all_b, window=11, n_out=200)
    x_up_b, y_up_b = smooth_curve(upper_b, window=5, n_out=80)
    x_lo_b, y_lo_b = smooth_curve(lower_b, window=5, n_out=80)

    # ================================================================
    # Save CSVs: for each figure, save the merged curve (best we can do
    # when solid+dashed overlap) plus the separated branches
    # ================================================================
    def save_digitized_csv(filename, x, y, label="signal"):
        path = os.path.join(OUTPUT_DIR, filename)
        with open(path, 'w') as f:
            f.write(f"Volume_mL,{label}\n")
            for i in range(len(x)):
                f.write(f"{x[i]:.4f},{y[i]:.4f}\n")
        print(f"  Saved: {path}")

    if len(x_all_a) > 0:
        save_digitized_csv("effio_fig3a_merged.csv", x_all_a, y_all_a, "UV_mAU")
    if len(x_up_a) > 0:
        save_digitized_csv("effio_fig3a_upper.csv", x_up_a, y_up_a, "UV_mAU")
    if len(x_lo_a) > 0:
        save_digitized_csv("effio_fig3a_lower.csv", x_lo_a, y_lo_a, "UV_mAU")

    if len(x_all_b) > 0:
        save_digitized_csv("effio_fig3b_merged.csv", x_all_b, y_all_b, "Cond_mScm")
    if len(x_up_b) > 0:
        save_digitized_csv("effio_fig3b_upper.csv", x_up_b, y_up_b, "Cond_mScm")
    if len(x_lo_b) > 0:
        save_digitized_csv("effio_fig3b_lower.csv", x_lo_b, y_lo_b, "Cond_mScm")

    # ================================================================
    # Diagnostic plots
    # ================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].imshow(arr3a)
    # Draw plot bbox
    t, b, l, r = bbox_3a
    axes[0, 0].plot([l, r, r, l, l], [t, t, b, b, t], 'r-', linewidth=1)
    axes[0, 0].set_title("Fig 3a: Crop + bbox")

    axes[0, 1].plot(x_all_a, y_all_a, 'b-', linewidth=2, label='Merged', alpha=0.7)
    if len(x_up_a) > 0:
        axes[0, 1].plot(x_up_a, y_up_a, 'r-', linewidth=1.5, label='Upper branch')
    if len(x_lo_a) > 0:
        axes[0, 1].plot(x_lo_a, y_lo_a, 'g-', linewidth=1.5, label='Lower branch')
    axes[0, 1].set_xlabel('Volume [mL]')
    axes[0, 1].set_ylabel('UV 215 nm [mAU]')
    axes[0, 1].set_title('Fig 3a: Digitized')
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].set_xlim(0, 3.5)
    axes[0, 1].set_ylim(0, 35)
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].imshow(arr3b)
    t, b, l, r = bbox_3b
    axes[1, 0].plot([l, r, r, l, l], [t, t, b, b, t], 'r-', linewidth=1)
    axes[1, 0].set_title("Fig 3b: Crop + bbox")

    axes[1, 1].plot(x_all_b, y_all_b, 'k-', linewidth=2, label='Merged', alpha=0.7)
    if len(x_up_b) > 0:
        axes[1, 1].plot(x_up_b, y_up_b, 'r-', linewidth=1.5, label='Upper branch')
    if len(x_lo_b) > 0:
        axes[1, 1].plot(x_lo_b, y_lo_b, 'g-', linewidth=1.5, label='Lower branch')
    axes[1, 1].set_xlabel('Volume [mL]')
    axes[1, 1].set_ylabel('Conductivity [mS/cm]')
    axes[1, 1].set_title('Fig 3b: Digitized')
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].set_xlim(0, 6)
    axes[1, 1].set_ylim(0, 6)
    axes[1, 1].grid(True, alpha=0.3)

    diag_path = os.path.join(OUTPUT_DIR, "effio_digitized_diagnostic.png")
    plt.tight_layout()
    fig.savefig(diag_path, dpi=150)
    plt.close(fig)
    print(f"\n  Saved diagnostic: {diag_path}")


if __name__ == '__main__':
    main()
