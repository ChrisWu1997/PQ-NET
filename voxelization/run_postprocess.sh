#!/bin/bash

# $1 is the input folder, e.g. /home/megaBeast/Desktop/partnet_data/voxelized/Chair
# $2 is the output folder, e.g. /dev/data/partnet_data/wurundi/voxelized/Chair

start_time=`date --date='0 days ago' "+%Y-%m-%d %H:%M:%S"`

# collect h5 voxel files
python convert_h5_vox.py --src $1 --out $2

# fill inner region to make voxel solid
python fill_part_solid.py --src $2 --out $2_solid

# scale solid part voxel to 64^3
python rescale_part_vox.py --src $2_solid

finish_time=`date --date='0 days ago' "+%Y-%m-%d %H:%M:%S"`
duration=$(($(($(date +%s -d "$finish_time")-$(date +%s -d "$start_time")))))
echo "total time: $duration seconds"
