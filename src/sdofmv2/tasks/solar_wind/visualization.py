import numpy as np
import pandas as pd
import torch

import matplotlib
import matplotlib.pyplot as plt


def colorbar_list(AIA_wavelengths, HMI_components):
    cmaplist = [
        "sdoaia94",
        "sdoaia131",
        "sdoaia171",
        "sdoaia193",
        "sdoaia211",
        "sdoaia304",
        "sdoaia335",
        "sdoaia1600",
        "sdoaia1700",
        "sdoaia4500",
        "hmimag",
        "hmimag",
        "hmimag",
    ]
    wavelengths_components = [
        "94A",
        "131A",
        "171A",
        "193A",
        "211A",
        "304A",
        "335A",
        "1600A",
        "1700A",
        "4500A",
        "Bx",
        "By",
        "Bz",
    ]
    cmap_dict = dict(zip(wavelengths_components, cmaplist))
    cmap_list = []
    for wavelength in AIA_wavelengths:
        cmap_list.append(cmap_dict[wavelength])
    for component in HMI_components:
        cmap_list.append(cmap_dict[component])
    return cmap_list


def wsa_to_image(lat_sh, lon_sh, r_sun=200, ctr_pxl=256):
    """
    Returns position on SDOML image of PSP footpoint from WSA.
    lat_sh and lon_sh are Stonyhurst latitude and longitude in degrees.
    r_sun is radius of sun on image in units of pixels.
    ctr_pixel is the value of the center of each axis in units of pixels.
    """
    lat_sh = (2 * np.pi / 360.0) * lat_sh
    lon_sh = (2 * np.pi / 360.0) * lon_sh
    x_pxl = np.round(r_sun * np.cos(lat_sh) * np.sin(lon_sh) + ctr_pxl).astype(int)
    y_pxl = np.round(r_sun * np.sin(lat_sh) + ctr_pxl).astype(int)
    return x_pxl, y_pxl


def plot_sdoml(
    datamodule,
    condition=None,
    dataset_idx=None,
    times=None,
    n_samples=None,
    wavelengths=[
        "131A",
        "1600A",
        "1700A",
        "171A",
        "193A",
        "211A",
        "304A",
        "335A",
        "94A",
    ],
    components=["Bx", "By", "Bz"],
    wsa_footpoint=False,
    title=None,
):
    indexed_data = datamodule.aligndata.reset_index()
    if (
        (dataset_idx is None) & (times is None) & (condition is None)
        | (dataset_idx is not None) & (times is not None)
        | (dataset_idx is not None) & (condition is not None)
        | (condition is not None) & (times is not None)
    ):
        raise TypeError("Must specify only one of condition, dataset_idx, times.")
    if dataset_idx is not None:
        try:
            iter(dataset_idx)
            ds_idx = dataset_idx.copy()
        except TypeError:
            ds_idx = [dataset_idx.copy()]
        subset = indexed_data[ds_idx]
        subset = subset[(subset["lon_footpoint"].abs() < 85)]
    if times is not None:
        try:
            iter(times)
            times_idx = times.copy()
        except TypeError:
            times_idx = [times.copy()]
        subset = datamodule.aligndata.loc[times_idx, :]
        subset["time_sdo_loc_est"] = subset.index
        subset = subset.reset_index(drop=True)
        subset = subset[(subset["lon_footpoint"].abs() < 85)]
        if len(subset) == 0:  # Is there no data?
            fig, axes = plt.subplots(
                1, 1, figsize=(2 * len(subset), 2 * len(wavelengths))
            )
            axes.axis("off")
            # Add text at relative position (0-1 scale)
            axes.text(
                0.5,
                0.5,
                f"No sample",
                transform=axes.transAxes,
                ha="center",
                va="center",
            )
            plt.suptitle(title)
            return (fig, axes)
    if condition is not None:
        subset = indexed_data[condition & (indexed_data["lon_footpoint"].abs() < 85)]
    if n_samples is not None:
        subset = subset.sample(n_samples)

    fig, axes = plt.subplots(
        len(wavelengths) + len(components),
        len(subset),
        figsize=(2 * len(subset), 2 * len(wavelengths)),
    )
    if title is not None:
        plt.suptitle(title)
    colormaps = colorbar_list(datamodule.wavelengths, datamodule.components)
    for idx, i in enumerate(subset.index):
        imgs = [
            datamodule.aia_data[subset.loc[i, "year"].astype(int).astype(str)][
                wavelength
            ][subset.loc[i, "idx_" + wavelength].astype(int)]
            for wavelength in datamodule.wavelengths
        ]
        for component in datamodule.components:
            imgs.append(
                datamodule.hmi_data[subset.loc[i, "year"].astype(int).astype(str)][
                    component
                ][subset.loc[i, "idx_" + component].astype(int)]
            )
        time = subset.loc[i, "time_sdo_loc_est"]

        if wsa_footpoint:
            wsa_lon = subset.loc[i, "lon_footpoint"]
            wsa_lat = subset.loc[i, "lat_footpoint"]
            x_foot, y_foot = wsa_to_image(wsa_lat, wsa_lon)

        plot_title = time.strftime("%Y-%m-%d\n%H:%M:%S")
        axes[0, idx].set_title(f"{plot_title}")
        for jdx, img in enumerate(imgs):  # Plot the images down the column
            if jdx < len(datamodule.wavelengths):  # For AIA data
                axes[jdx, idx].imshow(
                    img, cmap=matplotlib.colormaps[colormaps[jdx]], norm="log"
                )
                axes[jdx, idx].text(
                    5, 482, wavelengths[jdx], color="white", size="x-small"
                )
            else:  # For HMI data
                dynamic_range = np.max(np.abs(img))
                axes[jdx, idx].imshow(
                    img,
                    cmap=matplotlib.colormaps[colormaps[jdx]],
                    vmin=-1 * dynamic_range,
                    vmax=dynamic_range,
                )
                axes[jdx, idx].text(
                    5,
                    482,
                    components[jdx - len(wavelengths)],
                    color="white",
                    size="x-small",
                )
            if wsa_footpoint:
                axes[jdx, idx].plot(
                    x_foot, y_foot, marker="x", color="k", markersize=5, alpha=0.3
                )
            axes[jdx, idx].axis("off")
    return (fig, axes)


def find_images_labels(
    imgs, timestamps, targets, preds, position, class_id, ch_id, max_samples=3
):
    """
    Args:
        imgs, targets, preds, position: Input tensors
        class_id: Class to filter for
        max_samples: Maximum number of samples to return (default: 3)
    """
    # Create masks for correct/incorrect predictions of the target class
    class_mask = targets == class_id
    correct_mask = torch.logical_and(targets == preds, class_mask)
    incorrect_mask = torch.logical_and(targets != preds, class_mask)

    correct_indices = torch.nonzero(correct_mask, as_tuple=True)[0]
    incorrect_indices = torch.nonzero(incorrect_mask, as_tuple=True)[0]

    def sample_indices(indices, max_samples):
        """Helper function to sample indices efficiently."""
        if len(indices) == 0:
            return torch.tensor([], dtype=torch.long, device=indices.device)
        elif len(indices) <= max_samples:
            return indices
        else:
            perm = torch.randperm(len(indices))[:max_samples]
            return indices[perm]

    # Sample indices
    chosen_correct_idx = sample_indices(correct_indices, max_samples)
    chosen_incorrect_idx = sample_indices(incorrect_indices, max_samples)

    # Create results
    def create_result_dict(chosen_idx, data_tensors, keys):
        """Helper to create result dictionary."""
        if len(chosen_idx) == 0:
            return {key: np.array([]) for key in keys}

        return {
            key: tensor[chosen_idx].detach().cpu().numpy()
            for key, tensor in zip(keys, data_tensors)
        }
        # result = {}
        # for key, tensor in zip(keys, data_tensors):
        #     print(key, chosen_idx)
        #     result[key] = tensor[chosen_idx].detach().cpu().numpy()

    keys = ["imgs", "timestamps", "targets", "preds", "position"]
    data_tensors = [imgs[:, ch_id, 0, :, :], timestamps, targets, preds, position]
    result_correct_dict = create_result_dict(chosen_correct_idx, data_tensors, keys)
    result_incorrect_dict = create_result_dict(chosen_incorrect_idx, data_tensors, keys)

    return result_correct_dict, result_incorrect_dict


def find_images_labels_embed(
    imgs, timestamps, targets, preds, position, class_id, max_samples=3
):
    """
    Args:
        imgs, targets, preds, position: Input tensors
        class_id: Class to filter for
        max_samples: Maximum number of samples to return (default: 3)
    """
    # Create masks for correct/incorrect predictions of the target class
    class_mask = targets == class_id
    correct_mask = torch.logical_and(targets == preds, class_mask)
    incorrect_mask = torch.logical_and(targets != preds, class_mask)

    correct_indices = torch.nonzero(correct_mask, as_tuple=True)[0]
    incorrect_indices = torch.nonzero(incorrect_mask, as_tuple=True)[0]

    def sample_indices(indices, max_samples):
        """Helper function to sample indices efficiently."""
        if len(indices) == 0:
            return torch.tensor([], dtype=torch.long, device=indices.device)
        elif len(indices) <= max_samples:
            return indices
        else:
            perm = torch.randperm(len(indices))[:max_samples]
            return indices[perm]

    # Sample indices
    chosen_correct_idx = sample_indices(correct_indices, max_samples)
    chosen_incorrect_idx = sample_indices(incorrect_indices, max_samples)

    # Create results
    def create_result_dict(chosen_idx, data_tensors, keys):
        """Helper to create result dictionary."""
        if len(chosen_idx) == 0:
            return {key: np.array([]) for key in keys}

        return {
            key: tensor[chosen_idx].detach().cpu().numpy()
            for key, tensor in zip(keys, data_tensors)
        }
        # result = {}
        # for key, tensor in zip(keys, data_tensors):
        #     print(key, chosen_idx)
        #     result[key] = tensor[chosen_idx].detach().cpu().numpy()

    keys = ["imgs", "timestamps", "targets", "preds", "position"]
    data_tensors = [imgs, timestamps, targets, preds, position]
    result_correct_dict = create_result_dict(chosen_correct_idx, data_tensors, keys)
    result_incorrect_dict = create_result_dict(chosen_incorrect_idx, data_tensors, keys)

    return result_correct_dict, result_incorrect_dict


def plot_images_grid(correct_data, plt_style, step=None):
    plt.style.use(plt_style)
    sdoaia193 = matplotlib.colormaps["sdoaia193"]
    fig, axes = plt.subplots(4, 3, figsize=(8, 12))

    for class_id in range(4):
        cor_imgs = correct_data["imgs"][class_id]
        if len(cor_imgs) == 0:
            # logger.info(f"Class {class_id} has no correct samples")
            for i in range(3):
                ax = axes[class_id, i]
                ax.axis("off")
                # Add text at relative position (0-1 scale)
                ax.text(
                    0.5,
                    0.5,
                    f"Class: {class_id}\nNo sample",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                )
                # ax.set_visible(False)
            continue

        gtuths = correct_data["targets"][class_id]
        pred_val = correct_data["preds"][class_id]
        loc = correct_data["position"][class_id]
        raw_times = correct_data["timestamps"][class_id]

        # Convert timestamps to UTC strings
        # times = [
        #     datetime.fromtimestamp(ts, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
        #     for ts in raw_times
        # ]
        times = pd.to_datetime(raw_times).strftime("%Y-%m-%d %H:%M:%S").to_list()

        num_images = cor_imgs.shape[0] if cor_imgs.ndim > 2 else 1

        for i in range(3):
            ax = axes[class_id, i]

            if i < num_images:
                if num_images >= 2:
                    img = cor_imgs[i]
                    current_gt = gtuths[i]
                    current_pred = pred_val[i]
                    current_loc = loc[i]
                    timestamp = times[i]
                else:
                    img = cor_imgs[0]
                    current_gt = gtuths[0]
                    current_pred = pred_val[0]
                    current_loc = loc[0]
                    timestamp = times[0]

                lon_psp = current_loc[0]
                lat_psp = current_loc[1]
                lat_footpoint = current_loc[2]
                lon_footpoint = current_loc[3]
                # lon_footpoint = current_loc[0]
                # lat_footpoint = current_loc[1]
                x_foot, y_foot = wsa_to_image(lat_footpoint, lon_footpoint)
                x_psp, y_psp = wsa_to_image(lat_psp, lon_psp)

                ax.imshow(img, cmap=sdoaia193)
                ax.plot(
                    x_foot,
                    y_foot,
                    marker="x",
                    color="white",
                    markersize=5,
                    label="footpoint",
                )
                ax.plot(
                    x_psp, y_psp, marker="x", color="grey", markersize=5, label="PSP"
                )
                ax.legend(
                    loc="upper center", ncols=2, fontsize="small", labelcolor="white"
                )
                ax.set_title(f"{timestamp}\nTarget: {current_gt}, Pred: {current_pred}")
                ax.axis("off")
            else:
                ax.axis("off")
                # Add text at relative position (0-1 scale)
                ax.text(
                    0.5,
                    0.5,
                    f"Class: {class_id}\nNo sample",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                )

    return fig


def plot_ecliptic(
    img,
    model,
    time,
    x_range=np.linspace(-213, 213, 30),
    y_range=np.linspace(-213, 213, 30),
    z_range=np.linspace(0, 0, 1),
    r_mean=np.float32(8.918359e07),
    r_std=np.float32(3.0130634e07),
    inner_mask=10,
    outer_mask=220,
):
    x_range_grid, y_range_grid, z_range_grid = np.meshgrid(x_range, y_range, z_range)

    psp_lat_grid = np.arctan2(
        z_range_grid, np.sqrt(x_range_grid**2 + y_range_grid**2)
    ) * (360.0 / (2 * np.pi))
    psp_lon_grid = np.arctan2(y_range_grid, x_range_grid) * (360.0 / (2 * np.pi))
    psp_r_grid = np.sqrt(
        x_range_grid**2 + y_range_grid**2 + z_range_grid**2
    )  # PSP radial position (must be normed before being passed to model)
    wsa_lat_grid = psp_lat_grid  # Solar disk latitudes
    wsa_lon_grid = psp_lon_grid  # Solar disk longitudes

    pos_arr = np.array(
        [
            psp_lon_grid.ravel(),  # Create position array (columns in order: PSP Lon, PSP, Lat, WSA Lat, WSA Lon)
            psp_lat_grid.ravel(),
            wsa_lat_grid.ravel(),
            wsa_lon_grid.ravel(),
        ]
    ).reshape(len(wsa_lon_grid.ravel()), 4)

    imgs_arr = np.zeros(
        (len(wsa_lat_grid.ravel()), 9, img.shape[1], img.shape[2], img.shape[3])
    )  # Make array of same image for each entry
    imgs_arr[:] = img[:9, :, :, :].numpy()

    pos_disk_grid = torch.from_numpy(pos_arr).to(
        torch.float32
    )  # Turn position array into torch tensor
    r_disk_grid = torch.from_numpy((psp_r_grid.ravel() * 695_700 - r_mean) / r_std).to(
        torch.float32
    )  # Make torch tensor from radial distance w/ normalization
    imgs_disk_grid = torch.from_numpy(imgs_arr).to(torch.float32)

    # Make the predictions in the ecliptic
    with torch.no_grad():
        dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        y_hat_disk_grid = model(
            imgs_disk_grid.to(dev), pos_disk_grid.to(dev), r_disk_grid.to(dev)
        )  # MAE only uses first 9 channels, that's why imgs is indexed the way it is
    preds_disk_grid = torch.argmax(y_hat_disk_grid, dim=1)

    # Reshape predictions into a grid
    preds_disk_grid_arr = preds_disk_grid.detach().cpu().numpy()

    # Plot predictions across the ecliptic
    fig, axes = plt.subplots(1, 1, figsize=(8, 8))
    timestr = pd.to_datetime(time).strftime(
        "%Y-%m-%d %H:%M:%S"
    )  # Could find another way to get first time

    preds_disk_grid_arr_masked = np.where(
        np.sqrt(x_range_grid**2 + y_range_grid**2) < outer_mask,
        preds_disk_grid_arr.reshape(x_range_grid.shape),
        np.nan,
    )

    # Plot classes as image in ecliptic
    labels = ["Ejecta", "Coronal Hole", "Sector Reversal", "Streamer Belt"]
    plt.imshow(
        preds_disk_grid_arr_masked[:, :, 0],
        cmap="viridis",
        vmin=0,
        vmax=3,
        origin="lower",
        extent=(x_range[0], x_range[-1], y_range[0], y_range[-1]),
        interpolation="gaussian",
    )
    cbar = plt.colorbar(label="SW Classes")
    cbar.set_ticks(ticks=[0, 1, 2, 3], labels=labels)

    # Add solar system objects/masks
    plt.scatter([0], [213], marker="o", color="k", label="Earth")
    sun_mask = plt.Circle((0, 0), inner_mask, color="gray")
    axes.add_patch(sun_mask)

    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # Adjust the axes:
    plt.xlim(-230, 230)
    plt.ylim(-230, 230)
    plt.gca().set_aspect("equal")

    axes.set_title(f"{timestr}\nSolar Wind Types in Ecliptic")
    axes.set_xlabel(r"Stonyhurst X ($R_S$)")
    axes.set_ylabel(r"Stonyhurst Y ($R_S$)")
    return fig


def plot_disk_distribution(
    img,
    model,
    time,
    wsa_lat=np.linspace(-75, 75, 30),
    wsa_lon=np.linspace(-90, 90, 30),
    labels=["Ejecta", "Coronal Hole", "Sector Reversal", "Streamer Belt"],
    colors=["#191923", "#0E79B2", "#bf1363", "#F39237"],
):
    psp_lat = 0 * wsa_lat  # PSP latitude (set constant)
    psp_lon = (wsa_lon + 180 - 5) % 360 - 180  # PSP longitude (set constant)
    psp_r = 0  # This is a scaled value. 0 = mean, 1 = 1std away from mean, etc.

    wsa_lat_grid, wsa_lon_grid = np.meshgrid(
        wsa_lat, wsa_lon
    )  # Latitude/longitude grid points (both 2D)
    psp_lat_grid, psp_lon_grid = np.meshgrid(psp_lat, psp_lon)
    pos_arr = np.array(
        [
            psp_lon_grid.ravel(),  # Create position array (columns in order: PSP Lon, PSP, Lat, WSA Lat, WSA Lon)
            psp_lat_grid.ravel(),
            wsa_lat_grid.ravel(),
            wsa_lon_grid.ravel(),
        ]
    ).reshape(len(wsa_lon_grid.ravel()), 4)

    imgs_arr = np.zeros(
        (len(wsa_lat_grid.ravel()), 9, img.shape[1], img.shape[2], img.shape[3])
    )  # Make array of same image for each entry
    imgs_arr[:] = img[:9, :, :, :].numpy()

    pos_disk_grid = torch.from_numpy(pos_arr).to(
        torch.float32
    )  # Turn position array into torch tensor
    r_disk_grid = torch.from_numpy(psp_r * np.ones(len(pos_arr))).to(
        torch.float32
    )  # Make torch tensor from radial distance
    imgs_disk_grid = torch.from_numpy(imgs_arr).to(torch.float32)
    # Make the predictions on the solar disk
    with torch.no_grad():
        dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        y_hat_disk_grid = model(
            imgs_disk_grid.to(dev), pos_disk_grid.to(dev), r_disk_grid.to(dev)
        )  # MAE only uses first 9 channels, that's why imgs is indexed the way it is
    preds_disk_grid = torch.argmax(y_hat_disk_grid, dim=1)

    # Reshape predictions into a grid
    preds_disk_grid_arr = preds_disk_grid.detach().cpu().numpy()
    sdoaia193 = matplotlib.colormaps["sdoaia193"]

    fig, axes = plt.subplots(1, 1, figsize=(8, 8))
    timestr = pd.to_datetime(time).strftime(
        "%Y-%m-%d %H:%M:%S"
    )  # Could find another way to get first time

    x_foot, y_foot = wsa_to_image(wsa_lat_grid.ravel(), wsa_lon_grid.ravel())

    # Plot the underlying solar disk
    axes.imshow(imgs_arr[0, 4, 0, :, :], cmap=sdoaia193)

    # Plot each class in turn
    for i in range(4):  # Loop over class labels
        indx = preds_disk_grid_arr == i
        plt.scatter(x_foot[indx], y_foot[indx], s=5, color=colors[i], label=labels[i])
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    axes.set_title(f"{timestr}\nSolar Disk Class Origin Distribution")
    axes.axis("off")
    return fig


def plot_tsne(
    embs,
    position,
    r_distance,
    labels,
    perplexities=[5, 30, 50, 100, 300],
    classes=["Ejecta", "Coronal Hole", "Sector Reversal", "Streamer Belt"],
    colors=["#191923", "#0E79B2", "#bf1363", "#F39237"],
):
    from sklearn import manifold
    from matplotlib.ticker import NullFormatter

    # Get everything into one tensor
    batch_size = embs.size(0)
    r_tensor = r_distance.to(dtype=embs.dtype, device=embs.device)
    # Ensure position encoding matches batch size
    if position.size(0) != batch_size:
        position = position.expand(batch_size, -1)
    # Concatenate and process
    r_tensor = r_tensor.view(batch_size, 1) if r_tensor.ndim == 1 else r_tensor
    combined = torch.cat([embs, position, r_tensor.reshape(batch_size, -1)], dim=-1)

    # Convert the tensor to numpy for sklearn
    X = combined.numpy()
    y = labels.numpy()

    # Generate t-SNE embeddings of the embeddings at various "perplexities"
    Y_list = []
    for i, perplexity in enumerate(perplexities):
        tsne = manifold.TSNE(
            n_components=2,
            init="random",
            random_state=0,
            perplexity=perplexity,
            max_iter=300,
        )
        Y = tsne.fit_transform(X)

    # Plot the clusters
    fig, axes = plt.subplots(1, len(perplexities), figsize=(25, 5))

    for i, perplexity in enumerate(perplexities):
        axes[i].set_title("Perplexity=%d" % perplexity)
        for j in range(4):  # Loop over class labels
            indx = y == j
            axes[i].scatter(
                Y_list[i][indx, 0],
                Y_list[i][indx, 1],
                color=colors[j],
                label=classes[j],
                alpha=0.25,
            )
        axes[i].xaxis.set_major_formatter(NullFormatter())
        axes[i].yaxis.set_major_formatter(NullFormatter())
        axes[i].axis("tight")

    return fig
