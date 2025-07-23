import os
from tqdm import tqdm
import shutil

from datasets import load_dataset

def sample_dataset(sample_target:list[int], data_dir, target_dir):
    for index, target_index in tqdm(enumerate(sample_target)):
        os.makedirs(f"{target_dir}/data/chunk-{index // 50:03d}", exist_ok=True)
        os.makedirs(
            f"{target_dir}/videos/chunk-{index // 50:03d}/observation.images.exo",
            exist_ok=True)
        os.makedirs(
            f"{target_dir}/videos/chunk-{index // 50:03d}/observation.images.wrist",
            exist_ok=True)
        os.makedirs(
            f"{target_dir}/videos/chunk-{index // 50:03d}/observation.images.table",
            exist_ok=True)

        parquet_src_file = f"{data_dir}/data/chunk-{target_index // 50:03d}/episode_{target_index:06d}.parquet"
        parquet_des_file = f"{target_dir}/data/chunk-{index // 50:03d}/episode_{index:06d}.parquet"

        exo_src_file = f"{data_dir}/videos/chunk-{target_index // 50:03d}/observation.images.exo/episode_{target_index:06d}.mp4"
        exo_des_file = f"{target_dir}/videos/chunk-{index // 50:03d}/observation.images.exo/episode_{index:06d}.mp4"

        wrist_src_file = f"{data_dir}/videos/chunk-{target_index // 50:03d}/observation.images.wrist/episode_{target_index:06d}.mp4"
        wrist_des_file = f"{target_dir}/videos/chunk-{index // 50:03d}/observation.images.wrist/episode_{index:06d}.mp4"

        table_src_file = f"{data_dir}/videos/chunk-{target_index // 50:03d}/observation.images.table/episode_{target_index:06d}.mp4"
        table_des_file = f"{target_dir}/videos/chunk-{index // 50:03d}/observation.images.table/episode_{index:06d}.mp4"

        src_dataset = load_dataset("parquet", data_files=parquet_src_file)['train']
        des_dataset = src_dataset.map(lambda x: {"episode_index": index})
        des_dataset.to_parquet(parquet_des_file)

        shutil.copy(exo_src_file, exo_des_file)
        shutil.copy(wrist_src_file, wrist_des_file)
        shutil.copy(table_src_file, table_des_file)


if __name__ == '__main__':
    sample_target = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590]
    data_dir = '/data/piper_grape0626/lerobot_5hz'
    target_dir = '/data/piper_grape0626_10'
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(os.path.join(target_dir, 'data'), exist_ok=True)
    os.makedirs(os.path.join(target_dir, 'meta'), exist_ok=True)
    os.makedirs(os.path.join(target_dir, 'videos'), exist_ok=True)

    sample_dataset(sample_target, data_dir, target_dir)

    sample_target = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136, 138, 140, 142, 144, 146, 148, 150, 152, 154, 156, 158, 160, 162, 164, 166, 168, 170, 172, 174, 176, 178, 180, 182, 184, 186, 188, 190, 192, 194, 196, 198, 200, 202, 204, 206, 208, 210, 212, 214, 216, 218, 220, 222, 224, 226, 228, 230, 232, 234, 236, 238, 240, 242, 244, 246, 248, 250, 252, 254, 256, 258, 260, 262, 264, 266, 268, 270, 272, 274, 276, 278, 280, 282, 284, 286, 288, 290, 292, 294, 296, 298, 300, 302, 304, 306, 308, 310, 312, 314, 316, 318, 320, 322, 324, 326, 328, 330, 332, 334, 336, 338, 340, 342, 344, 346, 348, 350, 352, 354, 356, 358, 360, 362, 364, 366, 368, 370, 372, 374, 376, 378, 380, 382, 384, 386, 388, 390, 392, 394, 396, 398, 400, 402, 404, 406, 408, 410, 412, 414, 416, 418, 420, 422, 424, 426, 428, 430, 432, 434, 436, 438, 440, 442, 444, 446, 448, 450, 452, 454, 456, 458, 460, 462, 464, 466, 468, 470, 472, 474, 476, 478, 480, 482, 484, 486, 488, 490, 492, 494, 496, 498, 500, 502, 504, 506, 508, 510, 512, 514, 516, 518, 520, 522, 524, 526, 528, 530, 532, 534, 536, 538, 540, 542, 544, 546, 548, 550, 552, 554, 556, 558, 560, 562, 564, 566, 568, 570, 572, 574, 576, 578, 580, 582, 584, 586, 588, 590, 592, 594, 596, 598]
    data_dir = '/data/piper_grape0626/lerobot_5hz'
    target_dir = '/data/piper_grape0626_50'
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(os.path.join(target_dir,'data'), exist_ok=True)
    os.makedirs(os.path.join(target_dir,'meta'), exist_ok=True)
    os.makedirs(os.path.join(target_dir,'videos'), exist_ok=True)

    sample_dataset(sample_target, data_dir, target_dir)

    sample_target = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22, 24, 25, 26, 28, 29, 30, 32, 33, 34, 36, 37, 38, 40, 41, 42, 44, 45, 46, 48, 49, 50, 52, 53, 54, 56, 57, 58, 60, 61, 62, 64, 65, 66, 68, 69, 70, 72, 73, 74, 76, 77, 78, 80, 81, 82, 84, 85, 86, 88, 89, 90, 92, 93, 94, 96, 97, 98, 100, 101, 102, 104, 105, 106, 108, 109, 110, 112, 113, 114, 116, 117, 118, 120, 121, 122, 124, 125, 126, 128, 129, 130, 132, 133, 134, 136, 137, 138, 140, 141, 142, 144, 145, 146, 148, 149, 150, 152, 153, 154, 156, 157, 158, 160, 161, 162, 164, 165, 166, 168, 169, 170, 172, 173, 174, 176, 177, 178, 180, 181, 182, 184, 185, 186, 188, 189, 190, 192, 193, 194, 196, 197, 198, 200, 201, 202, 204, 205, 206, 208, 209, 210, 212, 213, 214, 216, 217, 218, 220, 221, 222, 224, 225, 226, 228, 229, 230, 232, 233, 234, 236, 237, 238, 240, 241, 242, 244, 245, 246, 248, 249, 250, 252, 253, 254, 256, 257, 258, 260, 261, 262, 264, 265, 266, 268, 269, 270, 272, 273, 274, 276, 277, 278, 280, 281, 282, 284, 285, 286, 288, 289, 290, 292, 293, 294, 296, 297, 298, 300, 301, 302, 304, 305, 306, 308, 309, 310, 312, 313, 314, 316, 317, 318, 320, 321, 322, 324, 325, 326, 328, 329, 330, 332, 333, 334, 336, 337, 338, 340, 341, 342, 344, 345, 346, 348, 349, 350, 352, 353, 354, 356, 357, 358, 360, 361, 362, 364, 365, 366, 368, 369, 370, 372, 373, 374, 376, 377, 378, 380, 381, 382, 384, 385, 386, 388, 389, 390, 392, 393, 394, 396, 397, 398, 400, 401, 402, 404, 405, 406, 408, 409, 410, 412, 413, 414, 416, 417, 418, 420, 421, 422, 424, 425, 426, 428, 429, 430, 432, 433, 434, 436, 437, 438, 440, 441, 442, 444, 445, 446, 448, 449, 450, 452, 453, 454, 456, 457, 458, 460, 461, 462, 464, 465, 466, 468, 469, 470, 472, 473, 474, 476, 477, 478, 480, 481, 482, 484, 485, 486, 488, 489, 490, 492, 493, 494, 496, 497, 498, 500, 501, 502, 504, 505, 506, 508, 509, 510, 512, 513, 514, 516, 517, 518, 520, 521, 522, 524, 525, 526, 528, 529, 530, 532, 533, 534, 536, 537, 538, 540, 541, 542, 544, 545, 546, 548, 549, 550, 552, 553, 554, 556, 557, 558, 560, 561, 562, 564, 565, 566, 568, 569, 570, 572, 573, 574, 576, 577, 578, 580, 581, 582, 584, 585, 586, 588, 589, 590, 592, 593, 594, 596, 597, 598]
    data_dir = '/data/piper_grape0626/lerobot_5hz'
    target_dir = '/data/piper_grape0626_75'
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(os.path.join(target_dir,'data'), exist_ok=True)
    os.makedirs(os.path.join(target_dir,'meta'), exist_ok=True)
    os.makedirs(os.path.join(target_dir,'videos'), exist_ok=True)

    sample_dataset(sample_target, data_dir, target_dir)