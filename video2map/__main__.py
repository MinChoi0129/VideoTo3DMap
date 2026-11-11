"""python -m video2map <video> <out_dir> <config.yaml>   ->   out_dir/map.ply, poses.txt, loops.json"""
import sys

from . import VideoToMap

if len(sys.argv) != 4:
    sys.exit(__doc__)
v2m = VideoToMap(sys.argv[3]); m = v2m(sys.argv[1]); n = m.save(sys.argv[2], v2m.cfg["output"]["voxel"])
print("%s: %d points, %d keyframes, %d loop closures" % (sys.argv[2], n, len(m.frame_ids), len(m.loops)))
