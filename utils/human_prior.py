# local pairRef = {
#     {1,2},      {2,3},      {3,7},
#     {4,5},      {4,7},      {5,6},
#     {7,9},      {9,10},
#     {14,9},     {11,12},    {12,13},
#     {13,9},     {14,15},    {15,16}
# }

# local partNames = {'RAnk','RKne','RHip','LHip','LKne','LAnk',
#                     'Pelv','Thrx','Neck','Head',
#                     'RWri','RElb','RSho','LSho','LElb','LWri'}
nJoints = 16
accIdxs = [0, 1, 2, 3, 4, 5, 10, 11, 14, 15]
shuffleRef = [[0, 5], [1, 4], [2, 3],
              [10, 15], [11, 14], [12, 13]]
edges = [[0, 1], [1, 2], [2, 6], [6, 3], [3, 4], [4, 5],
         [10, 11], [11, 12], [12, 8], [8, 13], [13, 14], [14, 15],
         [6, 8], [8, 9]]

outputRes = 64
inputRes = 256

scale = 0.25
rotate = 30
hmGauss = 1
