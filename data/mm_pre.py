from torch.utils.data import Dataset
import torch
import numpy as np

__all__ = ['MMDataset']

class MMDataset(Dataset):
    """Dataset wrapper for aligned text, video, and audio features."""

    def __init__(self, label_ids, text_data, video_data, audio_data, indexes=None):
        self.label_ids = label_ids
        self.text_data = text_data
        self.video_data = video_data
        self.audio_data = audio_data
        self.indexes = indexes
        self.size = len(self.text_data)

        # Use contiguous IDs to index per-sample weights.
        self.row_indexes = np.arange(self.size, dtype=np.int64)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        sample = {
            'text_feats': torch.tensor(self.text_data[index]),
            'video_feats': torch.tensor(np.array(self.video_data['feats'][index])),
            'video_lengths': torch.tensor(np.array(self.video_data['lengths'][index])),
            'audio_feats': torch.tensor(np.array(self.audio_data['feats'][index])),
            'audio_lengths': torch.tensor(np.array(self.audio_data['lengths'][index])),
        }
        if self.label_ids is not None:
            sample['label_ids'] = torch.tensor(self.label_ids[index])

        sample['indexes'] = int(self.row_indexes[index])

        if self.indexes is not None:
            sample['raw_index'] = self.indexes[index]

        return sample
