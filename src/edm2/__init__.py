import sys
import pickle
from edm2 import dnnlib, torch_utils, training

# HACK edm model were pickled and hence retain the original repo folder structure
# here, manually re-route the imports
sys.modules["torch_utils"] = torch_utils
sys.modules["dnnlib"] = dnnlib
sys.modules["training"] = training

edm2_nvidia_root = "https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions"

EDM_MODELS = {
    "edm2-img512-xs-fid": dnnlib.EasyDict(
        net=f"{edm2_nvidia_root}/edm2-img512-xs-2147483-0.135.pkl"
    ),  # fid = 3.53
    "edm2-img512-s-fid": dnnlib.EasyDict(
        net=f"{edm2_nvidia_root}/edm2-img512-s-2147483-0.130.pkl"
    ),  # fid = 2.56
    "edm2-img512-m-fid": dnnlib.EasyDict(
        net=f"{edm2_nvidia_root}/edm2-img512-m-2147483-0.100.pkl"
    ),  # fid = 2.25
    "edm2-img512-l-fid": dnnlib.EasyDict(
        net=f"{edm2_nvidia_root}/edm2-img512-l-1879048-0.085.pkl"
    ),  # fid = 2.06
    "edm2-img512-xl-fid": dnnlib.EasyDict(
        net=f"{edm2_nvidia_root}/edm2-img512-xl-1342177-0.085.pkl"
    ),  # fid = 1.96
    "edm2-img512-xxl-fid": dnnlib.EasyDict(
        net=f"{edm2_nvidia_root}/edm2-img512-xxl-0939524-0.070.pkl"
    ),  # fid = 1.91
    "edm2-img64-s-fid": dnnlib.EasyDict(
        net=f"{edm2_nvidia_root}/edm2-img64-s-1073741-0.075.pkl"
    ),  # fid = 1.58
    "edm2-img64-m-fid": dnnlib.EasyDict(
        net=f"{edm2_nvidia_root}/edm2-img64-m-2147483-0.060.pkl"
    ),  # fid = 1.43
    "edm2-img64-l-fid": dnnlib.EasyDict(
        net=f"{edm2_nvidia_root}/edm2-img64-l-1073741-0.040.pkl"
    ),  # fid = 1.33
    "edm2-img64-xl-fid": dnnlib.EasyDict(
        net=f"{edm2_nvidia_root}/edm2-img64-xl-0671088-0.040.pkl"
    ),  # fid = 1.33
    "edm2-img512-xs-dino": dnnlib.EasyDict(
        net=f"{edm2_nvidia_root}/edm2-img512-xs-2147483-0.200.pkl"
    ),  # fd_dinov2 = 103.39
    "edm2-img512-s-dino": dnnlib.EasyDict(
        net=f"{edm2_nvidia_root}/edm2-img512-s-2147483-0.190.pkl"
    ),  # fd_dinov2 = 68.64
    "edm2-img512-m-dino": dnnlib.EasyDict(
        net=f"{edm2_nvidia_root}/edm2-img512-m-2147483-0.155.pkl"
    ),  # fd_dinov2 = 58.44
    "edm2-img512-l-dino": dnnlib.EasyDict(
        net=f"{edm2_nvidia_root}/edm2-img512-l-1879048-0.155.pkl"
    ),  # fd_dinov2 = 52.25
    "edm2-img512-xl-dino": dnnlib.EasyDict(
        net=f"{edm2_nvidia_root}/edm2-img512-xl-1342177-0.155.pkl"
    ),  # fd_dinov2 = 45.96
    "edm2-img512-xxl-dino": dnnlib.EasyDict(
        net=f"{edm2_nvidia_root}/edm2-img512-xxl-0939524-0.150.pkl"
    ),  # fd_dinov2 = 42.84
    "edm2-img512-xs-guid-fid": dnnlib.EasyDict(
        net=f"{edm2_nvidia_root}/edm2-img512-xs-2147483-0.045.pkl",
        gnet=f"{edm2_nvidia_root}/edm2-img512-xs-uncond-2147483-0.045.pkl",
        guidance=1.4,
    ),  # fid = 2.91
    "edm2-img512-s-guid-fid": dnnlib.EasyDict(
        net=f"{edm2_nvidia_root}/edm2-img512-s-2147483-0.025.pkl",
        gnet=f"{edm2_nvidia_root}/edm2-img512-xs-uncond-2147483-0.025.pkl",
        guidance=1.4,
    ),  # fid = 2.23
    "edm2-img512-m-guid-fid": dnnlib.EasyDict(
        net=f"{edm2_nvidia_root}/edm2-img512-m-2147483-0.030.pkl",
        gnet=f"{edm2_nvidia_root}/edm2-img512-xs-uncond-2147483-0.030.pkl",
        guidance=1.2,
    ),  # fid = 2.01
    "edm2-img512-l-guid-fid": dnnlib.EasyDict(
        net=f"{edm2_nvidia_root}/edm2-img512-l-1879048-0.015.pkl",
        gnet=f"{edm2_nvidia_root}/edm2-img512-xs-uncond-2147483-0.015.pkl",
        guidance=1.2,
    ),  # fid = 1.88
    "edm2-img512-xl-guid-fid": dnnlib.EasyDict(
        net=f"{edm2_nvidia_root}/edm2-img512-xl-1342177-0.020.pkl",
        gnet=f"{edm2_nvidia_root}/edm2-img512-xs-uncond-2147483-0.020.pkl",
        guidance=1.2,
    ),  # fid = 1.85
    "edm2-img512-xxl-guid-fid": dnnlib.EasyDict(
        net=f"{edm2_nvidia_root}/checkpoints/edm2-img512-xxl-0939524-0.015.pkl",
        gnet=f"{edm2_nvidia_root}/edm2-img512-xs-uncond-2147483-0.015.pkl",
        guidance=1.2,
    ),  # fid = 1.81
    "edm2-img512-xs-guid-dino": dnnlib.EasyDict(
        net=f"{edm2_nvidia_root}/edm2-img512-xs-2147483-0.150.pkl",
        gnet=f"{edm2_nvidia_root}/edm2-img512-xs-uncond-2147483-0.150.pkl",
        guidance=1.7,
    ),  # fd_dinov2 = 79.94
    "edm2-img512-s-guid-dino": dnnlib.EasyDict(
        net=f"{edm2_nvidia_root}/edm2-img512-s-2147483-0.085.pkl",
        gnet=f"{edm2_nvidia_root}/edm2-img512-xs-uncond-2147483-0.085.pkl",
        guidance=1.9,
    ),  # fd_dinov2 = 52.32
    "edm2-img512-m-guid-dino": dnnlib.EasyDict(
        net=f"{edm2_nvidia_root}/edm2-img512-m-2147483-0.015.pkl",
        gnet=f"{edm2_nvidia_root}/edm2-img512-xs-uncond-2147483-0.015.pkl",
        guidance=2.0,
    ),  # fd_dinov2 = 41.98
    "edm2-img512-l-guid-dino": dnnlib.EasyDict(
        net=f"{edm2_nvidia_root}/edm2-img512-l-1879048-0.035.pkl",
        gnet=f"{edm2_nvidia_root}/edm2-img512-xs-uncond-2147483-0.035.pkl",
        guidance=1.7,
    ),  # fd_dinov2 = 38.20
    "edm2-img512-xl-guid-dino": dnnlib.EasyDict(
        net=f"{edm2_nvidia_root}/edm2-img512-xl-1342177-0.030.pkl",
        gnet=f"{edm2_nvidia_root}/edm2-img512-xs-uncond-2147483-0.030.pkl",
        guidance=1.7,
    ),  # fd_dinov2 = 35.67
    "edm2-img512-xxl-guid-dino": dnnlib.EasyDict(
        net=f"{edm2_nvidia_root}/edm2-img512-xxl-0939524-0.015.pkl",
        gnet=f"{edm2_nvidia_root}/edm2-img512-xs-uncond-2147483-0.015.pkl",
        guidance=1.7,
    ),  # fd_dinov2 = 33.09
}


def load_edm2(network_url, verbose: bool = True, device: str = "cpu"):
    # Load main network.
    if isinstance(network_url, str):
        if verbose:
            print(f"Loading network from {network_url} ...")
        with dnnlib.util.open_url(network_url, verbose=verbose) as f:
            data = pickle.load(f)
        net = data["ema"].to(device)

        encoder = data.get("encoder", None)

        if encoder is None:
            # The models provided by "Kynkäänniemi, Tuomas" don't come with an encoder
            if "-phema-" in network_url:
                config_xs = EDM_MODELS["edm2-img512-xs-fid"]
                with dnnlib.util.open_url(config_xs["net"], verbose=verbose) as f:
                    data_xs = pickle.load(f)

                encoder = data_xs["encoder"]
            else:
                encoder = dnnlib.util.construct_class_by_name(
                    class_name="training.encoders.StandardRGBEncoder"
                )
        return net, encoder
    raise ValueError("network_id must be a string")
