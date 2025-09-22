# Garment Support (Deprecated)

In RoboVerse, we also integrate garment and deformable object manipulation tasks into the RoboVerse benchmark. We build upon the foundations of [GarmentLab](https://github.com/GarmentLab/GarmentLab). While we initially attempted to migrate these tasks as-is from the [main](https://github.com/RoboVerseOrg/RoboVerse) branch, we encountered an incompatibility issue with the PhysX simulation backend between [OmniIsaacGymEnvs](https://github.com/isaac-sim/OmniIsaacGymEnvs) and the provided garment simulation. So, garment manipulation currently is supported on [`garment`](https://github.com/RoboVerseOrg/RoboVerse/tree/garment) branch. As the community’s research on soft body control in simulations progresses, we will continue to update this branch.

## 🛠️ Get Started

Follow the steps below to get the Garment Support feature up and running:

### 1. Install Isaac Sim 4.0.0

This branch is built with Isaac Sim 4.0.0, which is required due to a [deprecation warning](https://docs.omniverse.nvidia.com/isaacsim/latest/archived_release_notes.html) for particle cloth in Isaac Sim 4.1.0 (used in the [main branch](https://github.com/RoboVerseOrg/RoboVerse)).

For convenience, you can add the following alias to your `.bashrc` or `.zshrc` to use the correct Python environment:

```shell
alias omni_python="~/.local/share/ov/pkg/isaac-sim-4.0.0/python.sh"
```


### 2. Download the Garment Assets

Make sure to download the required [GarmentLab assets](https://drive.google.com/drive/folders/1CqJILIK8VQ-RCuLa_aFN-WtYTbovpFga) and extract them into the `Assets` folder within your project directory. Ensure that the folder structure matches the original asset directory and that all files are correctly placed.

### 3. Install the Additional Dependencies

Once the assets are in place, you’ll need to install some additional dependencies. Run the following command:

```shell
omni_python -m pip install open3d
```

### 4. Launch the Demo Collection

The demo scripts are located in the `scripts` folder. You can launch the demo with the following command:

```shell
omni_python demo/HangDemo.py --usd_path $usd_path --visual_material_path $visual_material_path
```

Then, the rendered data will be stored in the same location as in the [main](https://github.com/RoboVerseOrg/RoboVerse) branch.

---

💡 **Notes**:
- **`usd_path`**: The path to the USD file for the garment (e.g., `Assets/Garment/Dress/.../garment.usd`).
- **`visual_material_path`**: The path to the material file for visual textures (e.g., `Assets/Material/.../material.usd`).

As we progress with garment manipulation research, we will continue updating this repository and resolving existing issues.

---

### 🛠️ Troubleshooting

If you encounter any issues with the setup, please open an issue in the repository. We appreciate your feedback as we continue to refine and improve this integration!

---

Feel free to ask if you need further assistance or details! 😊
