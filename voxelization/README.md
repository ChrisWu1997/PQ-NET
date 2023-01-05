# Voxelization

For those who need to train the model on their own dataset, here is our voxelization process.

1. Download and compile [this mesh voxelization library](https://github.com/davidstutz/mesh-voxelization), and put the executable file `voxelize` under this directory.

2. Organize the PartNet mesh data as follows, e.g., 
   ```shell
   Bed/
      45/ # shape id
         00.obj # part mesh, possibly not watertight
         01.obj
         ...
         04.obj
      ...
      10967/
         00.obj
         01.obj
         ...
         16.obj
      ...
   ```
   It's important to name each obj file as `xx.obj`(xx being the index of part) as it is required when running voxelization. 

3. Convert .obj to .off and scale the shapes from `[-1, 1]^3` to `[0, 64]^3`.
   ```shell
   python obj2off_scaled.py -i Bed -o Bed_off
   ```

4. Run voxelization. `run_voxelize.sh`. This will voxelize each shape folder (containing part `.obj` files) into a single h5 file (containting voxels of dimension `(n_parts, 64, 64, 64)`).

   ```shell
   bash run_voxelize.sh Bed_off Bed_off_h5 0 64
   ```

5. Run post-processing. This will post-process the above h5 files, e.g. fill inside voxels and scale each individual part to a $64^3$ canonical space.

   ```bash
   bash run_postprocess.sh Bed_off_h5 Bed_off_h5_post
   ```
   After the program finishes, `Bed_off_h5_post` will contain h5 data that can be used for training.
