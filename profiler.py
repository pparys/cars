import time
import json
import psutil
import torch
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List
import sys
import logging

logger = logging.getLogger(__name__)

@dataclass
class TrieStats:
    total_nodes: int = 0
    max_depth: int = 0
    total_children: int = 0
    avg_branching_factor: float = 0.0
    nodes_per_depth: Dict[int, int] = None
    
    def __post_init__(self):
        if self.nodes_per_depth is None:
            self.nodes_per_depth = {}
    
    def to_dict(self):
        return asdict(self)


@dataclass
class MemoryStats:
    trie_cpu_bytes: int = 0
    trie_gpu_bytes: int = 0
    model_gpu_bytes: int = 0
    system_ram_mb: float = 0.0
    gpu_allocated_mb: float = 0.0
    gpu_reserved_mb: float = 0.0
    
    def to_dict(self):
        return asdict(self)


@dataclass
class ComputeStats:
    total_inference_time: float = 0.0
    total_logits_processing_time: float = 0.0
    total_trie_operations_time: float = 0.0
    num_forward_passes: int = 0
    num_trie_lookups: int = 0
    num_trie_insertions: int = 0
    num_recomputations: int = 0
    
    def to_dict(self):
        return asdict(self)


@dataclass
class SamplingStats:
    total_samples_attempted: int = 0
    successful_samples: int = 0
    failed_samples: int = 0
    total_tokens_generated: int = 0
    avg_tokens_per_sample: float = 0.0
    avg_raw_logprob: float = 0.0
    avg_constrained_logprob: float = 0.0
    
    equivalent_llm_tokens: int = 0
    trie_reuse_rate: float = 0.0  
    
    def to_dict(self):
        return asdict(self)


class Profiler:
    def __init__(self, log_dir: str, enabled: bool = True):
        self.log_dir = log_dir
        self.enabled = enabled
        
        self.trie_stats = TrieStats()
        self.memory_stats = MemoryStats()
        self.compute_stats = ComputeStats()
        self.sampling_stats = SamplingStats()
        
        self.step_profiles: List[Dict] = []
        self.start_time = None
        self.last_checkpoint_time = None
        
    def start(self):
        if not self.enabled:
            return
        self.start_time = time.time()
        self.last_checkpoint_time = self.start_time
        self._capture_initial_memory()
        
    def _capture_initial_memory(self):
        if not self.enabled:
            return
        
        process = psutil.Process()
        self.memory_stats.system_ram_mb = process.memory_info().rss / (1024 * 1024)
        
        if torch.cuda.is_available():
            self.memory_stats.gpu_allocated_mb = torch.cuda.memory_allocated() / (1024 * 1024)
            self.memory_stats.gpu_reserved_mb = torch.cuda.memory_reserved() / (1024 * 1024)
    
    def profile_trie(self, trie):
        if not self.enabled:
            return
        
        stats = TrieStats()
        nodes_per_depth = {}
        
        def traverse(node, depth=0):
            stats.total_nodes += 1
            stats.max_depth = max(stats.max_depth, depth)
            stats.total_children += len(node.children)
            
            if depth not in nodes_per_depth:
                nodes_per_depth[depth] = 0
            nodes_per_depth[depth] += 1
            
            for child in node.children.values():
                traverse(child, depth + 1)
        
        traverse(trie.root)
        
        if stats.total_nodes > 0:
            stats.avg_branching_factor = stats.total_children / stats.total_nodes
        stats.nodes_per_depth = nodes_per_depth
        
        self.trie_stats = stats
        
        self._compute_trie_memory(trie)
        
    def _compute_trie_memory(self, trie):
        if not self.enabled:
            return
        
        cpu_bytes = 0
        gpu_bytes = 0
        
        def traverse(node):
            nonlocal cpu_bytes, gpu_bytes
            
            # Base object size
            cpu_bytes += sys.getsizeof(node)
            cpu_bytes += sys.getsizeof(node.children)
            
            # Tensors
            if node.raw_logprob is not None:
                tensor_bytes = node.raw_logprob.element_size() * node.raw_logprob.nelement()
                if node.raw_logprob.is_cuda:
                    gpu_bytes += tensor_bytes
                else:
                    cpu_bytes += tensor_bytes
                    
            if node.log_theta is not None:
                tensor_bytes = node.log_theta.element_size() * node.log_theta.nelement()
                if node.log_theta.is_cuda:
                    gpu_bytes += tensor_bytes
                else:
                    cpu_bytes += tensor_bytes
            
            for child in node.children.values():
                traverse(child)
        
        traverse(trie.root)
        
        self.memory_stats.trie_cpu_bytes = cpu_bytes
        self.memory_stats.trie_gpu_bytes = gpu_bytes
    
    def record_sample_attempt(self, success: bool, num_tokens: int = 0, 
                             raw_logprob: float = 0.0, cons_logprob: float = 0.0):
        if not self.enabled:
            return
        
        self.sampling_stats.total_samples_attempted += 1
        if success:
            self.sampling_stats.successful_samples += 1
            self.sampling_stats.total_tokens_generated += num_tokens
            self.sampling_stats.avg_raw_logprob += raw_logprob
            self.sampling_stats.avg_constrained_logprob += cons_logprob
        else:
            self.sampling_stats.failed_samples += 1
    
    def record_inference_time(self, time_seconds: float):
        if not self.enabled:
            return
        self.compute_stats.total_inference_time += time_seconds
        self.compute_stats.num_forward_passes += 1
    
    def record_logits_processing_time(self, time_seconds: float):
        if not self.enabled:
            return
        self.compute_stats.total_logits_processing_time += time_seconds
    
    def record_trie_operation(self, operation_type: str, time_seconds: float = 0.0):
        if not self.enabled:
            return
        
        self.compute_stats.total_trie_operations_time += time_seconds
        
        if operation_type == 'lookup':
            self.compute_stats.num_trie_lookups += 1
        elif operation_type == 'insert':
            self.compute_stats.num_trie_insertions += 1
        elif operation_type == 'recompute':
            self.compute_stats.num_recomputations += 1
    
    def checkpoint(self, step: int, trie=None):
        if not self.enabled:
            return
        
        current_time = time.time()
        elapsed = current_time - self.last_checkpoint_time
        
        checkpoint_data = {
            'step': step,
            'timestamp': current_time,
            'elapsed_since_last': elapsed,
            'sampling_stats': self.sampling_stats.to_dict(),
            'compute_stats': self.compute_stats.to_dict(),
        }
        
        if trie is not None:
            self.profile_trie(trie)
            checkpoint_data['trie_stats'] = self.trie_stats.to_dict()
            checkpoint_data['memory_stats'] = self.memory_stats.to_dict()
        
        self.step_profiles.append(checkpoint_data)
        self.last_checkpoint_time = current_time
    
    def finalize(self, trie=None):
        if not self.enabled:
            return {}
        
        if trie is not None:
            self.profile_trie(trie)
        
        if self.sampling_stats.successful_samples > 0:
            self.sampling_stats.avg_tokens_per_sample = (
                self.sampling_stats.total_tokens_generated / 
                self.sampling_stats.successful_samples
            )
            self.sampling_stats.avg_raw_logprob /= self.sampling_stats.successful_samples
            self.sampling_stats.avg_constrained_logprob /= self.sampling_stats.successful_samples
        
        if self.compute_stats.num_trie_lookups > 0:
            hits = self.compute_stats.num_trie_lookups - self.compute_stats.num_trie_insertions
            self.sampling_stats.trie_reuse_rate = hits / self.compute_stats.num_trie_lookups

        self.sampling_stats.equivalent_llm_tokens = (
            self.sampling_stats.total_tokens_generated +
            self.sampling_stats.failed_samples * 50 # Assumption
        )
        
        self._capture_initial_memory()
        
        total_time = time.time() - self.start_time if self.start_time else 0
        
        return {
            'total_time': total_time,
            'trie_stats': self.trie_stats.to_dict(),
            'memory_stats': self.memory_stats.to_dict(),
            'compute_stats': self.compute_stats.to_dict(),
            'sampling_stats': self.sampling_stats.to_dict(),
            'checkpoints': self.step_profiles,
        }
    
    def save(self, filename: str = "profile.json", trie=None):
        if not self.enabled:
            return
        
        results = self.finalize(trie)
        
        filepath = f"{self.log_dir}/{filename}"
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Profiling results saved to {filepath}")
        self._save_summary(results)
    
    def _save_summary(self, results):
        summary_path = f"{self.log_dir}/profile_summary.txt"
        
        with open(summary_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("CARS PROFILING SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("TRIE STATISTICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total nodes: {self.trie_stats.total_nodes}\n")
            f.write(f"Max depth: {self.trie_stats.max_depth}\n")
            f.write(f"Avg branching factor: {self.trie_stats.avg_branching_factor:.2f}\n")
            f.write(f"Nodes per depth: {self.trie_stats.nodes_per_depth}\n\n")
            
            f.write("MEMORY USAGE\n")
            f.write("-" * 80 + "\n")
            f.write(f"Trie CPU memory: {self.memory_stats.trie_cpu_bytes / (1024**2):.2f} MB\n")
            f.write(f"Trie GPU memory: {self.memory_stats.trie_gpu_bytes / (1024**2):.2f} MB\n")
            f.write(f"GPU allocated: {self.memory_stats.gpu_allocated_mb:.2f} MB\n")
            f.write(f"System RAM: {self.memory_stats.system_ram_mb:.2f} MB\n\n")
            
            f.write("COMPUTE STATISTICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total inference time: {self.compute_stats.total_inference_time:.2f} s\n")
            f.write(f"Total logits processing time: {self.compute_stats.total_logits_processing_time:.2f} s\n")
            f.write(f"Total trie operations time: {self.compute_stats.total_trie_operations_time:.2f} s\n")
            f.write(f"Number of forward passes: {self.compute_stats.num_forward_passes}\n")
            f.write(f"Trie lookups: {self.compute_stats.num_trie_lookups}\n")
            f.write(f"Trie insertions: {self.compute_stats.num_trie_insertions}\n")
            f.write(f"Recomputations: {self.compute_stats.num_recomputations}\n\n")
            
            f.write("SAMPLING STATISTICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total samples attempted: {self.sampling_stats.total_samples_attempted}\n")
            f.write(f"Successful samples: {self.sampling_stats.successful_samples}\n")
            f.write(f"Failed samples: {self.sampling_stats.failed_samples}\n")
            f.write(f"Success rate: {self.sampling_stats.successful_samples / max(1, self.sampling_stats.total_samples_attempted) * 100:.1f}%\n")
            f.write(f"Total tokens generated: {self.sampling_stats.total_tokens_generated}\n")
            f.write(f"Avg tokens per sample: {self.sampling_stats.avg_tokens_per_sample:.1f}\n")
            f.write(f"Avg raw logprob: {self.sampling_stats.avg_raw_logprob:.2f}\n")
            f.write(f"Avg constrained logprob: {self.sampling_stats.avg_constrained_logprob:.2f}\n")
            f.write(f"Trie reuse rate: {self.sampling_stats.trie_reuse_rate * 100:.1f}%\n\n")
            
            f.write("LLM COST COMPARISON\n")
            f.write("-" * 80 + "\n")
            f.write(f"Equivalent LLM tokens (estimated): {self.sampling_stats.equivalent_llm_tokens}\n")
            f.write(f"Trie memory overhead: {(self.memory_stats.trie_cpu_bytes + self.memory_stats.trie_gpu_bytes) / (1024**2):.2f} MB\n")
            
            total_time = results.get('total_time', 1)
            tps = self.sampling_stats.total_tokens_generated / total_time
            f.write(f"Tokens per second: {tps:.2f}\n")
            
            f.write("\n" + "=" * 80 + "\n")
        
        logger.info(f"Profile summary saved to {summary_path}")


def create_profiler(log_dir: str, enabled: bool = True) -> Profiler:
    return Profiler(log_dir, enabled)