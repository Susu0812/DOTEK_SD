# row anchors are a series of pre-defined coordinates in image height to detect lanes
# the row anchors are defined according to the evaluation protocol of CULane and Tusimple
# since our method will resize the image to 288x384 for training, the row anchors are defined with the height of 288
# you can modify these row anchors according to your training image resolution

tusimple_row_anchor = [ 64,  68,  72,  76,  80,  84,  88,  92,  96, 100, 104, 108, 112,
            116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164,
            168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216,
            220, 224, 228, 232, 236, 240, 244, 248, 252, 256, 260, 264, 268,
            272, 276, 280, 284]
culane_row_anchor = [121, 131, 141, 150, 160, 170, 180, 189, 199, 209, 219, 228, 238, 248, 258, 267, 277, 287]

my_row_anchor1 = [
    146, 149, 152, 155, 158, 161, 164, 167,
    170, 173, 176, 179, 182, 185, 188, 191,
    194, 197, 200, 203, 206, 209, 212, 215,
    218, 221, 224, 227, 230, 233, 236, 239,
    242, 245, 248, 251, 254, 257, 260, 263,
    266, 269, 272, 275, 278, 281, 284, 287
]
my_row_anchor2 = [149, 155, 161, 167, 173, 179, 185, 191,
 197, 203, 209, 215, 221, 227, 233, 239,
 245, 251, 257, 263, 269, 275, 281, 287]

my_row_anchor = [121, 131, 141, 150, 160, 170, 180, 189, 199, 209, 219, 228, 238, 248, 258, 267, 277, 287]

my_row_anchor3 = [
116, 122, 128, 134, 140, 146, 152, 158, 
164, 170, 176, 182, 188, 194, 200, 206, 
212, 217, 222, 227, 232, 237, 242, 247, 
252, 257, 262, 267, 272, 277, 282, 287]