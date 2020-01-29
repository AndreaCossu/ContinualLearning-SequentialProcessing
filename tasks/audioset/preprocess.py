import h5py
import numpy as np
import torch

filtered_eval = [4,6,10,525,14,15,17,18,21,23,24,25,33,34,38,43,45,49,
53,54,56,58,59,60,62,63,67,70,77,79,82,92,94,96,105,149,
174,182,183,184,198,201,203,207,209,211,215,292,309,312,317,318,327,339,
342,346,347,348,361,367,368,369,377,382,386,387,390,391,398,400,403,407,
411,414,415,419,421,442,450,452,453,454,467,468,469,470,471,481,483,493,
498,500]



eval_no_restr_no_child = [1, 2, 3, 4, 5, 6, 517, 10, 12, 525, 14, 15, 13, 17, 18,
21, 23, 24, 25, 28, 33, 34, 36, 38, 43, 45, 49, 53, 54, 56, 58, 59, 60, 62, 63, 66,
67, 68, 70, 75, 77, 79, 82, 83, 91, 92, 94, 96, 100, 101, 105, 107, 113, 116, 127, 133,
141, 142, 143, 146, 147, 149, 151, 156, 157, 159, 163, 168, 170, 172, 174, 180, 181,
182, 183, 184, 187, 188, 193, 194, 196, 198, 199, 201, 203, 207, 209, 210, 211,
212, 215, 285, 287, 291, 292, 295, 304, 309, 312, 315, 317, 318, 323, 325, 326,
327, 331, 332, 334, 339, 340, 342, 346, 347, 348, 352, 353, 361, 367, 368, 369,
370, 371, 374, 377, 382, 386, 387, 390, 391, 397, 398, 400, 403, 407, 411, 414,
415, 419, 421, 428, 430, 442, 450, 452, 453, 454, 467, 468, 469, 470, 471, 476,
481, 483, 493, 498, 500, 506, 507, 509, 510]

# total balanced set: torch.Size([22160, 10, 128])
# filtered with above list: torch.Size([1576, 10, 128])

def bool_to_float32(y):
    return np.float32(y)

def uint8_to_float32(x):
    return (np.float32(x) - 128.) / 128.

def load_all_data(hdf5_path):
    with h5py.File(hdf5_path, 'r') as hf:
        x = hf.get('x')
        y = hf.get('y')
        video_id_list = hf.get('video_id_list')
        x = np.array(x)
        y = list(y)
        video_id_list = list(video_id_list)

        x = uint8_to_float32(x)		# shape: (N, 10, 128)
        y = bool_to_float32(y)		# shape: (N, 527)

    return torch.from_numpy(x), torch.from_numpy(y).long(), video_id_list

def select_category(ids, labels, one_category=False):
    '''
    Select only input from category id (0,..., 526)

    If one_category is True the selection is restricted only
    to those inputs with only 1 category marked
    '''

    # 0 and 137 from balanced have many samples

    # select rows only with 1 one
    if one_category:
        idx_one = labels.sum(dim=1) == 1
        if (idx_one == 0).all().item() == 1:
            return None
    else:
        # identity for &
        idx_one = torch.ones(labels.size(0), dtype=torch.uint8)

    # select rows with at least the correct category
    if type(ids) != list:
        idx = labels[:,ids] == 1
    # select rows with at least the correct categories
    else:
        idx = (labels[:,ids].sum(dim=1) == 1)

    final_idx = idx & idx_one
    if (final_idx==0).all().item() == 1:
        return None
    else:
        return final_idx


def generator_audioset(hdf5_path, block_size=10000):
    with h5py.File(hdf5_path, 'r') as f:

        # f.keys() = 'video_id_list', 'x', 'y'
        # len of file = 2041789 ~2M
        total_len = 2041789
        start = 0
        for stop in range(block_size, total_len, block_size):
            x = f['x'][start:stop]
            y = f['y'][start:stop]
            video_id_list = f['video_id_list'][start:stop]
            x = np.array(x)
            y = list(y)
            video_id_list = list(video_id_list)

            x = uint8_to_float32(x) # shape: (N, 10, 128)
            y = bool_to_float32(y) # shape: (N, 527)
            start += block_size

            yield torch.from_numpy(x), torch.from_numpy(y).long(), video_id_list


if __name__ == '__main__':

    base = 'tasks/audioset/data/packed_features/'
    filename = 'bal_train.h5'
    filename2 = 'unbal_train.h5'


    # x of shape (B, 10, 128) # 128 embeddings for each of the 10 seconds
    # y of shape (B, 527) # 527 classes
    x, y, ids = load_all_data(base+filename)

    '''
    data = select_category(137, y, False)
    data2 = select_category(0, y, False)
    data.tolist()
    data2.tolist()

    i=0value
    for a,b in zip(data, data2):
        if a.item() == 1 and b.item() == 1:
            print(i)
        i +=1
    '''

    x, labels, _ = load_all_data(base+filename)
    print(x.size())
    data = select_category(eval_no_restr_no_child, labels, one_category=True)
    print(data.nonzero().size())
