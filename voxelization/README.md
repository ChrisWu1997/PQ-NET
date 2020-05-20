# Voxelization

For those who need to train the model on their own dataset, here is our voxelization process.

1. Download and compile [this mesh voxelization library](https://github.com/davidstutz/mesh-voxelization), and put the executable file `voxelize` under this directory.

2. Organize the part mesh (in global frame) files like: `‘{class_name}/{shape_id}/object_{n}.obj’`, `n = 1,2,…,k` and k is the total number of parts.

3. Run `run_voxelize.sh`. This will voxelize each mesh file into h5 file.

   ```bash
   ./run_voxelize.sh {input_folder} {output_folder} 0 {voxel resolution}
   ```

4. Run `run_postprocess.sh`. This will post-process the above h5 file,  e.g. make voxels solid and scale each part to $64^3$.

   ```bash
   ./run_postprocess.sh {input_h5_folder} {output_h5_folder}
   ```

   

