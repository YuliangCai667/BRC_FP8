import gymnasium as gym
import numpy as np

import os
import pickle
import json

from jaxrl.utils import Batch

class ParallelReplayBuffer:
    def __init__(self, observation_space: gym.spaces.Box, action_dim: int, capacity: int, num_tasks: int):
        self.observations = np.empty((num_tasks, capacity, observation_space.shape[-1]), dtype=observation_space.dtype)
        self.actions = np.empty((num_tasks, capacity, action_dim), dtype=np.float32)
        self.rewards = np.empty((num_tasks, capacity, ), dtype=np.float32)
        self.masks = np.empty((num_tasks, capacity, ), dtype=np.float32)
        self.next_observations = np.empty((num_tasks, capacity, observation_space.shape[-1]), dtype=observation_space.dtype)
        self.size = 0
        self.insert_index = 0
        self.capacity = capacity
        self.n_parts = 4
        self.num_tasks = num_tasks

    def insert(self, observation: np.ndarray, action: np.ndarray, reward: float, mask: float, next_observation: np.ndarray):
        self.observations[:, self.insert_index] = observation
        self.actions[:, self.insert_index] = action
        self.rewards[:, self.insert_index] = reward
        self.masks[:, self.insert_index] = mask
        self.next_observations[:, self.insert_index] = next_observation
        self.insert_index = (self.insert_index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int, num_batches: int):
        indx = np.random.randint(self.size * self.num_tasks, size=(num_batches, batch_size))
        task_indx, sample_indx = np.divmod(indx, self.size)
        observations = self.observations[task_indx, sample_indx, :]
        actions = self.actions[task_indx, sample_indx, :]
        rewards = self.rewards[task_indx, sample_indx]
        masks = self.masks[task_indx, sample_indx]
        next_observations = self.next_observations[task_indx, sample_indx, :]
        return Batch(observations=observations,
                     actions=actions,
                     rewards=rewards,
                     masks=masks,
                     next_observations=next_observations,
                     task_ids=task_indx)    

    def sample_task_batches(self):
        batch_size = 32
        indxs = np.random.randint(self.size, size=batch_size)        
        task_ids = np.zeros((self.num_tasks, batch_size), dtype=np.int32) + np.arange(self.num_tasks, dtype=np.int32)[:, None]
        return Batch(observations=self.observations[:, indxs],
                     actions=self.actions[:, indxs],
                     rewards=self.rewards[:, indxs],
                     masks=self.masks[:, indxs],
                     next_observations=self.next_observations[:, indxs],
                     task_ids=task_ids)

    def make_probe_batch(self, batch_size: int = 256, seed: int = 0):
        """Sample with a private RNG so diagnostics do not perturb training RNG."""
        if self.size == 0:
            raise ValueError("cannot create a probe batch from an empty replay buffer")
        rng = np.random.RandomState(seed)
        indices = rng.randint(self.size * self.num_tasks, size=int(batch_size))
        task_indices, sample_indices = np.divmod(indices, self.size)
        return Batch(
            observations=self.observations[task_indices, sample_indices, :].copy(),
            actions=self.actions[task_indices, sample_indices, :].copy(),
            rewards=self.rewards[task_indices, sample_indices].copy(),
            masks=self.masks[task_indices, sample_indices].copy(),
            next_observations=self.next_observations[task_indices, sample_indices, :].copy(),
            task_ids=task_indices.astype(np.int32),
        )

    def estimate_size_bytes(self) -> int:
        valid_size = self.capacity if self.size == self.capacity else self.size
        arrays = [self.observations, self.actions, self.rewards, self.masks, self.next_observations]
        return int(sum(array[:, :valid_size].nbytes for array in arrays))

    def save(self, save_dir: str, target_chunk_bytes: int = 256 * 1024 * 1024):
        """Save only valid slots, bounding each temporary array slice."""
        os.makedirs(save_dir, exist_ok=True)
        arrays = {
            'observations': self.observations,
            'actions': self.actions,
            'rewards': self.rewards,
            'masks': self.masks,
            'next_observations': self.next_observations,
        }
        valid_size = self.capacity if self.size == self.capacity else self.size
        bytes_per_slot = max(sum(array[:, :1].nbytes for array in arrays.values()), 1)
        chunk_size = max(1, int(target_chunk_bytes) // bytes_per_slot)
        chunks = []
        for chunk_id, start in enumerate(range(0, valid_size, chunk_size)):
            end = min(start + chunk_size, valid_size)
            chunk_files = {}
            for name, array in arrays.items():
                filename = f'{name}_{chunk_id:06d}.npy'
                np.save(os.path.join(save_dir, filename), array[:, start:end], allow_pickle=False)
                chunk_files[name] = filename
            chunks.append({'start': start, 'end': end, 'files': chunk_files})
        manifest = {
            'schema_version': 1,
            'capacity': self.capacity,
            'num_tasks': self.num_tasks,
            'size': self.size,
            'insert_index': self.insert_index,
            'valid_size': valid_size,
            'chunks': chunks,
        }
        with open(os.path.join(save_dir, 'manifest.json'), 'w', encoding='utf-8') as file:
            json.dump(manifest, file, indent=2)
        return manifest

    def load(self, save_dir: str):
        manifest_path = os.path.join(save_dir, 'manifest.json')
        if not os.path.exists(manifest_path):
            return self._load_legacy(save_dir)
        with open(manifest_path, encoding='utf-8') as file:
            manifest = json.load(file)
        if manifest['capacity'] != self.capacity or manifest['num_tasks'] != self.num_tasks:
            raise ValueError('replay buffer checkpoint shape/configuration mismatch')
        arrays = {
            'observations': self.observations,
            'actions': self.actions,
            'rewards': self.rewards,
            'masks': self.masks,
            'next_observations': self.next_observations,
        }
        for chunk in manifest['chunks']:
            start, end = chunk['start'], chunk['end']
            for name, array in arrays.items():
                data = np.load(os.path.join(save_dir, chunk['files'][name]), allow_pickle=False)
                expected_shape = array[:, start:end].shape
                if data.shape != expected_shape or data.dtype != array.dtype:
                    raise ValueError(f'invalid replay chunk for {name}: {data.shape}/{data.dtype}')
                array[:, start:end] = data
        self.size = int(manifest['size'])
        self.insert_index = int(manifest['insert_index'])
        return manifest

    def _load_legacy(self, save_dir: str):
        data_path = os.path.join(save_dir, 'buffer')
        chunk_size = self.capacity // self.n_parts
        for i in range(self.n_parts):
            data_path_splitted = data_path.split('buffer')
            data_path_splitted[-1] = f'_chunk_{i}{data_path_splitted[-1]}'
            data_path_chunk = 'buffer'.join(data_path_splitted)
            data_chunk = pickle.load(open(data_path_chunk, 'rb'))
            self.observations[:, i*chunk_size:(i+1)*chunk_size], \
                self.actions[:, i*chunk_size:(i+1)*chunk_size], \
                self.rewards[:, i*chunk_size:(i+1)*chunk_size], \
                self.masks[:, i*chunk_size:(i+1)*chunk_size], \
                self.next_observations[:, i*chunk_size:(i+1)*chunk_size] = data_chunk
        with open(os.path.join(save_dir, 'buffer_info'), 'rb') as file:
            self.size, self.insert_index = pickle.load(file)
