#!/bin/bash

# $1 is the input folder.
# $2 is the output folder.
# $3 is the number of shape to read. use 0 to process all
# $4 is the resolution of voxel

# /home/megaBeast/Desktop/partnet_data/partMesh/Chair/  /home/megaBeast/Desktop/partnet_data/voxelized/Chair/ 0 64

start_time=`date --date='0 days ago' "+%Y-%m-%d %H:%M:%S"`

mkdir -p $2

shape_num=1

for element in `ls $1`
    do
        dir_or_file=$1/$element
        if [ -d $dir_or_file ]
        then
            echo "+++++ Read "$shape_num" shape in "$dir_or_file" +++++"
            tiny_start_time=`date --date='0 days ago' "+%Y-%m-%d %H:%M:%S"`

            if [ $shape_num -gt 0 ]
            then
                ./voxelize occ $1/$element/ $2/$element.h5 --width $4 --height $4 --depth $4
            fi

            tiny_finish_time=`date --date='0 days ago' "+%Y-%m-%d %H:%M:%S"`
            tiny_duration=$(($(($(date +%s -d "$tiny_finish_time")-$(date +%s -d "$tiny_start_time")))))
            echo "Shape $shape_num use $tiny_duration seconds"

            if [ $shape_num -eq $3 ]
            then
                break
            fi
            shape_num=`expr $shape_num + 1`

        fi
    done

finish_time=`date --date='0 days ago' "+%Y-%m-%d %H:%M:%S"`
duration=$(($(($(date +%s -d "$finish_time")-$(date +%s -d "$start_time")))))
echo "total time: $duration seconds"
