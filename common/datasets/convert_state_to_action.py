from datasets import load_dataset
from tqdm import tqdm

def shift_action_to_action_fk(dataset):
    # change col name from 'action' to 'action_fk'
    dataset = dataset.rename_column('action', 'action_fk')
    return dataset


def copy_state_to_action(dataset):
    # copy col 'observation.state' and paste to col 'action'
    dataset = dataset.add_column('action', dataset['observation.state'])
    return dataset


def convert_state_to_action(index):
    parquet_file_path = f"/data/piper_lerobot/lerobot_aligncups_5hz/train/data/chunk-{index//50:03d}/episode_{index:06d}.parquet"
    parquet_file_path_des = f"/data/piper_lerobot/lerobot_aligncups_5hz/train/data/chunk-{index//50:03d}/episode_{index:06d}.parquet"

    dataset = load_dataset("parquet", data_files=parquet_file_path)['train']

    if 'action_fk' not in dataset.column_names:
        dataset = shift_action_to_action_fk(dataset)
        dataset = copy_state_to_action(dataset)

        dataset.to_parquet(parquet_file_path_des)


if __name__ == "__main__":
    for i in tqdm(range(120,200)):
        convert_state_to_action(i)