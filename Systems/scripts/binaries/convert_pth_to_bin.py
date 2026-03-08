import torch
import numpy as np
from sys import argv

if __name__ == "__main__":
    if len(argv) != 2:
        print("USAGE: python convert_pth_to_bin.py <path to .pth>")
        exit(1)

    state = torch.load(argv[1], map_location="cpu")

    print(state.keys())

    def save_tensor(tensor, filename):
        arr = tensor.detach().cpu().numpy().astype(np.float32)
        arr.tofile(filename)
        print(f"wrote {filename}  shape={arr.shape}")

    def save_bn(prefix, layer_idx):
        gamma = state[f"features.{layer_idx}.weight"]
        beta  = state[f"features.{layer_idx}.bias"]
        mean  = state[f"features.{layer_idx}.running_mean"]
        var   = state[f"features.{layer_idx}.running_var"]

        packed = np.concatenate([
            gamma.numpy(),
            beta.numpy(),
            mean.numpy(),
            var.numpy()
        ]).astype(np.float32)

        packed.tofile(prefix)
        print(f"wrote {prefix}  shape={(4, gamma.numel())}")


    # Layer 1
    save_tensor(state["features.0.weight"], "weights_conv_1d_l1.bin")
    save_tensor(state["features.0.bias"],   "weights_conv_1d_bias_l1.bin")
    save_bn("weights_bn_1d_l1.bin", 1)

    # Layer 2
    save_tensor(state["features.4.weight"], "weights_conv_1d_l2.bin")
    save_tensor(state["features.4.bias"],   "weights_conv_1d_bias_l2.bin")
    save_bn("weights_bn_1d_l2.bin", 5)

    # Layer 3
    save_tensor(state["features.8.weight"], "weights_conv_1d_l3.bin")
    save_tensor(state["features.8.bias"],   "weights_conv_1d_bias_l3.bin")
    save_bn("weights_bn_1d_l3.bin", 9)

    # Layer 4
    save_tensor(state["features.11.weight"], "weights_conv_1d_l4.bin")
    save_tensor(state["features.11.bias"],   "weights_conv_1d_bias_l4.bin")
    save_bn("weights_bn_1d_l4.bin", 12)

    # ---- Linear Layers ----

    save_tensor(state["classifier.3.weight"], "linear_mat_1.bin")
    save_tensor(state["classifier.3.bias"],   "linear_add_1.bin")

    save_tensor(state["classifier.6.weight"], "linear_mat_2.bin")
    save_tensor(state["classifier.6.bias"],   "linear_add_2.bin")

    print("\nDone.")
