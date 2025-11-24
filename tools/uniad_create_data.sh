
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python tools/create_data.py nuscenes --root-path ./data/nuscenes \
       --out-dir ./data/infos \
       --extra-tag nuscenes \
       --version v1.0 \
       --canbus ./data/nuscenes \


# python tools/create_data.py nuscenes --root-path /home/xueshu/work/data/nuscenes/data/nuscenes \
#        --out-dir /home/xueshu/work/data/nuscenes/data/infos \
#        --extra-tag nuscenes \
#        --version v1.0-mini \
#        --canbus /home/xueshu/work/data/nuscenes/data/nuscenes \