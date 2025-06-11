#!/usr/bin/env python3
"""
FIXED SplatFlow Dataset Cache Manager
Properly detects existing caches and manages dataset downloads.
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import hashlib
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from datasets import load_dataset, get_dataset_config_names, get_dataset_split_names
    from datasets.utils.info_utils import VerificationMode
    from datasets import config as datasets_config
    from transformers import GPT2Tokenizer
    DATASETS_AVAILABLE = True
    print("✅ Datasets and transformers available")
except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    print("Install with: pip install datasets transformers")
    sys.exit(1)


class FixedSplatFlowDatasetManager:
    """FIXED dataset manager with proper cache detection"""
    
    def __init__(self):
        # Get the actual datasets cache directory
        self.cache_dir = Path(datasets_config.HF_DATASETS_CACHE)
        self.tokenizer = None
        
        logger.info(f"📁 Using datasets cache directory: {self.cache_dir}")
        
        # SplatFlow dataset configurations (FIXED with proper detection)
        self.splatflow_datasets = {
            "roneneldan/TinyStories": {
                "name": "TinyStories",
                "description": "High-quality children's stories",
                "split": "train",
                "text_field": "text", 
                "target_count": 3000,
                "streaming": True,
                "config": None,
                "priority": 1  # Highest priority
            },
            "squad": {
                "name": "SQuAD",
                "description": "High-quality Wikipedia paragraphs", 
                "split": "train",
                "text_field": "context",
                "target_count": 2000,
                "streaming": False,
                "config": None,
                "priority": 2
            },
            "ag_news": {
                "name": "AG News",
                "description": "News articles",
                "split": "train", 
                "text_field": "text",
                "target_count": 1500,
                "streaming": False,
                "config": None,
                "priority": 3
            },
            "cnn_dailymail": {
                "name": "CNN/DailyMail",
                "description": "Professional news articles",
                "split": "train",
                "text_field": "article", 
                "target_count": 2000,
                "streaming": True,
                "config": "3.0.0",
                "priority": 4
            },
            "openwebtext": {
                "name": "OpenWebText", 
                "description": "Curated web content",
                "split": "train",
                "text_field": "text",
                "target_count": 2500, 
                "streaming": True,
                "config": None,
                "priority": 5
            },
            "bookcorpus": {
                "name": "BookCorpus",
                "description": "Literature excerpts", 
                "split": "train",
                "text_field": "text",
                "target_count": 2000,
                "streaming": True,
                "config": None,
                "priority": 6
            },
            "multi_news": {
                "name": "MultiNews",
                "description": "Multi-document news",
                "split": "train", 
                "text_field": "document",
                "target_count": 800,
                "streaming": False,
                "config": None,
                "priority": 7
            }
        }
        
    def setup_tokenizer(self):
        """Setup tokenizer for validation"""
        try:
            logger.info("Setting up tokenizer...")
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("✅ Tokenizer ready")
        except Exception as e:
            logger.warning(f"Failed to setup tokenizer: {e}")
            self.tokenizer = None
    
    def _get_dataset_cache_key(self, dataset_id: str, config: Optional[str] = None, split: str = "train") -> str:
        """Generate the cache key that datasets library uses"""
        # This mimics how the datasets library generates cache keys
        
        if config:
            cache_key_parts = [dataset_id, config, split]
        else:
            cache_key_parts = [dataset_id, split]
            
        # Create a hash for the cache key (similar to datasets library)
        cache_string = "__".join(cache_key_parts)
        return cache_string
    
    def _find_dataset_cache_files(self, dataset_id: str, config: Optional[str] = None) -> Tuple[bool, List[Path]]:
        """FIXED: Properly find dataset cache files using datasets library methods"""
        
        try:
            # Method 1: Check if we can load the dataset info without downloading
            from datasets import load_dataset_builder
            
            builder_kwargs = {"path": dataset_id}
            if config:
                builder_kwargs["name"] = config
            
            try:
                builder = load_dataset_builder(**builder_kwargs)
                
                # Check if the dataset has any cached files
                cache_files = []
                
                # Look for cache files in the builder's cache directory
                if hasattr(builder, '_cache_dir') and builder._cache_dir:
                    cache_dir = Path(builder._cache_dir)
                    if cache_dir.exists():
                        cache_files = list(cache_dir.rglob("*.arrow"))
                        cache_files.extend(list(cache_dir.rglob("*.json")))
                        cache_files.extend(list(cache_dir.rglob("dataset_info.json")))
                
                # Alternative: Look in the general cache directory
                if not cache_files:
                    # Create a normalized dataset name for cache directory
                    normalized_name = dataset_id.replace("/", "___")
                    potential_cache_dirs = [
                        self.cache_dir / normalized_name,
                        self.cache_dir / dataset_id.replace("/", "--"),
                        self.cache_dir / f"{normalized_name}___default"
                    ]
                    
                    if config:
                        potential_cache_dirs.extend([
                            self.cache_dir / f"{normalized_name}___{config}",
                            self.cache_dir / f"{dataset_id.replace('/', '--')}--{config}"
                        ])
                    
                    for cache_dir in potential_cache_dirs:
                        if cache_dir.exists():
                            files = list(cache_dir.rglob("*.arrow"))
                            files.extend(list(cache_dir.rglob("*.json")))
                            if files:
                                cache_files.extend(files)
                                break
                
                is_cached = len(cache_files) > 0
                return is_cached, cache_files
                
            except Exception as e:
                logger.debug(f"Builder method failed for {dataset_id}: {e}")
                
        except Exception as e:
            logger.debug(f"Cache detection failed for {dataset_id}: {e}")
        
        # Method 2: Direct cache directory scanning
        return self._scan_cache_directories(dataset_id, config)
    
    def _scan_cache_directories(self, dataset_id: str, config: Optional[str] = None) -> Tuple[bool, List[Path]]:
        """Scan cache directories for dataset files"""
        
        if not self.cache_dir.exists():
            return False, []
        
        # Generate possible cache directory names
        normalized_name = dataset_id.replace("/", "___")
        possible_cache_names = [
            normalized_name,
            dataset_id.replace("/", "--"),
            dataset_id.replace("/", "_"),
            dataset_id.lower().replace("/", "___"),
            dataset_id.lower().replace("/", "--"),
            f"{normalized_name}___default",
            f"{normalized_name}__default",
        ]
        
        if config:
            config_variants = [
                f"{normalized_name}___{config}",
                f"{normalized_name}__{config}",
                f"{dataset_id.replace('/', '--')}--{config}",
                f"{normalized_name}___{config}___train",
            ]
            possible_cache_names.extend(config_variants)
        
        cache_files = []
        
        for cache_name in possible_cache_names:
            potential_path = self.cache_dir / cache_name
            if potential_path.exists():
                # Look for arrow files (dataset files) or json files (metadata)
                files = list(potential_path.rglob("*.arrow"))
                files.extend(list(potential_path.rglob("dataset_info.json"))
                files.extend(list(potential_path.rglob("*.json")))
                
                if files:
                    cache_files.extend(files)
                    logger.debug(f"Found cache files for {dataset_id} in {potential_path}")
        
        return len(cache_files) > 0, cache_files
    
    def check_cache_status(self, detailed: bool = False) -> Dict:
        """FIXED: Check which datasets are actually cached"""
        logger.info("🔍 Checking FIXED cache status...")
        
        cached = {}
        total_size_mb = 0
        
        for dataset_id, dataset_config in self.splatflow_datasets.items():
            name = dataset_config["name"]
            config = dataset_config.get("config")
            
            logger.debug(f"Checking cache for {name} ({dataset_id})...")
            
            is_cached, cache_files = self._find_dataset_cache_files(dataset_id, config)
            
            # Calculate cache size
            cache_size_mb = 0
            if cache_files:
                for file_path in cache_files:
                    try:
                        if file_path.exists():
                            cache_size_mb += file_path.stat().st_size / (1024 * 1024)
                    except Exception:
                        pass
                total_size_mb += cache_size_mb
            
            cached[dataset_id] = {
                "name": name,
                "is_cached": is_cached,
                "cache_files": [str(f) for f in cache_files] if detailed else len(cache_files),
                "cache_size_mb": cache_size_mb,
                "priority": dataset_config["priority"],
                "config": config
            }
            
            status = "✅ CACHED" if is_cached else "❌ NOT CACHED"
            size_str = f"({cache_size_mb:.1f}MB)" if cache_size_mb > 0 else ""
            logger.info(f"   {name}: {status} {size_str}")
            
            if detailed and cache_files:
                for file_path in cache_files[:3]:  # Show first 3 files
                    logger.info(f"      📄 {file_path}")
                if len(cache_files) > 3:
                    logger.info(f"      ... and {len(cache_files) - 3} more files")
        
        cached_count = sum(1 for info in cached.values() if info["is_cached"])
        total_count = len(cached)
        
        logger.info(f"📊 Cache summary: {cached_count}/{total_count} datasets cached")
        logger.info(f"💾 Total cache size: {total_size_mb:.1f}MB ({total_size_mb/1024:.2f}GB)")
        
        return {
            "datasets": cached,
            "summary": {
                "cached_count": cached_count,
                "total_count": total_count,
                "cache_percentage": (cached_count / total_count) * 100,
                "total_size_mb": total_size_mb,
                "total_size_gb": total_size_mb / 1024
            }
        }
    
    def estimate_download_size(self) -> Tuple[float, Dict]:
        """FIXED: Better download size estimation"""
        # More accurate estimates based on actual dataset sizes
        size_estimates_mb = {
            "roneneldan/TinyStories": 2400,    # ~2.4GB
            "squad": 35,                       # ~35MB  
            "ag_news": 30,                     # ~30MB
            "cnn_dailymail": 1100,             # ~1.1GB
            "openwebtext": 38000,              # ~38GB (very large!)
            "bookcorpus": 4800,                # ~4.8GB  
            "multi_news": 1200                 # ~1.2GB
        }
        
        total_mb = sum(size_estimates_mb.values())
        total_gb = total_mb / 1024
        
        return total_gb, size_estimates_mb
    
    def preload_dataset(self, dataset_id: str, config: Dict, force_reload: bool = False, 
                       verify_only: bool = False) -> bool:
        """FIXED: Pre-load a single dataset with better error handling"""
        name = config["name"]
        
        if verify_only:
            logger.info(f"🔍 Verifying {name}...")
        else:
            logger.info(f"📥 Pre-loading {name}...")
            
        logger.info(f"    Dataset: {dataset_id}")
        logger.info(f"    Description: {config['description']}")
        logger.info(f"    Config: {config.get('config', 'default')}")
        logger.info(f"    Split: {config['split']}")
        logger.info(f"    Streaming: {config['streaming']}")
        
        try:
            start_time = time.time()
            
            # Build load arguments
            load_kwargs = {
                "path": dataset_id,
                "split": config["split"],
                "streaming": config["streaming"]
            }
            
            if config["config"]:
                load_kwargs["name"] = config["config"]
            
            # Add verification mode to speed up checking
            if verify_only:
                load_kwargs["verification_mode"] = VerificationMode.NO_CHECKS
            
            logger.debug(f"Loading with kwargs: {load_kwargs}")
            
            # Load dataset
            dataset = load_dataset(**load_kwargs)
            
            # Sample and validate data
            text_field = config["text_field"]
            target_count = min(config["target_count"], 50 if verify_only else 200)
            
            sampled_count = 0
            valid_texts = 0
            sample_texts = []
            
            logger.info(f"    📊 Sampling {target_count} items for validation...")
            
            for item in dataset:
                if sampled_count >= target_count:
                    break
                
                try:
                    text = item.get(text_field, "")
                    
                    # Handle list fields (like in some datasets)
                    if isinstance(text, list) and len(text) > 0:
                        text = text[0] if len(text) == 1 else " | ".join([t for t in text if t and len(t.strip()) > 20])
                    
                    if isinstance(text, str) and len(text.strip()) > 50:
                        valid_texts += 1
                        
                        # Store sample for quality check
                        if len(sample_texts) < 3:
                            sample_texts.append(text[:200] + "..." if len(text) > 200 else text)
                        
                        # Quick tokenization test if tokenizer available
                        if self.tokenizer and sampled_count < 5:
                            try:
                                tokens = self.tokenizer.encode(text[:500], max_length=100, truncation=True)
                                if len(tokens) < 5:
                                    logger.warning(f"Very short tokenization for {name}")
                            except Exception:
                                pass
                    
                    sampled_count += 1
                    
                    # Progress update
                    if sampled_count % 25 == 0:
                        logger.info(f"      Processed {sampled_count}/{target_count} samples...")
                        
                except Exception as e:
                    logger.warning(f"Error processing item in {name}: {e}")
                    continue
            
            load_time = time.time() - start_time
            
            # Validation check
            if valid_texts == 0:
                logger.error(f"    ❌ {name} validation failed - no valid texts found!")
                return False
            
            validation_rate = valid_texts / max(sampled_count, 1) * 100
            
            logger.info(f"    ✅ {name} {'verified' if verify_only else 'loaded'} successfully!")
            logger.info(f"       Processed: {sampled_count} samples")
            logger.info(f"       Valid texts: {valid_texts} ({validation_rate:.1f}%)")
            logger.info(f"       Time: {load_time:.1f}s")
            
            # Show sample texts for verification
            if sample_texts and not verify_only:
                logger.info(f"       Sample texts:")
                for i, sample in enumerate(sample_texts):
                    logger.info(f"         {i+1}. {sample}")
            
            return True
            
        except Exception as e:
            logger.error(f"    ❌ Failed to {'verify' if verify_only else 'load'} {name}: {e}")
            return False
    
    def preload_selected_datasets(self, dataset_names: List[str] = None, 
                                force_reload: bool = False, skip_cached: bool = True,
                                verify_only: bool = False) -> Dict:
        """Pre-load selected datasets or all datasets"""
        
        action = "verification" if verify_only else "pre-loading"
        
        if dataset_names:
            logger.info(f"🚀 Starting FIXED SplatFlow dataset {action} for selected datasets...")
            
            # Filter to requested datasets
            selected_datasets = {k: v for k, v in self.splatflow_datasets.items() 
                               if v["name"] in dataset_names or k in dataset_names}
            
            if not selected_datasets:
                logger.error(f"No datasets found matching: {dataset_names}")
                return {"success": [], "failed": [], "skipped": []}
        else:
            logger.info(f"🚀 Starting FIXED SplatFlow dataset {action} for all datasets...")
            selected_datasets = self.splatflow_datasets
        
        if not verify_only:
            total_gb, size_estimates = self.estimate_download_size()
            logger.info(f"   Estimated download size: {total_gb:.1f}GB")
        
        logger.info(f"   Force reload: {force_reload}")
        logger.info(f"   Skip cached: {skip_cached}")
        
        # Check current cache status
        cache_status = self.check_cache_status()
        
        # Setup tokenizer for validation
        if not verify_only:
            self.setup_tokenizer()
        
        results = {
            "success": [],
            "failed": [],
            "skipped": []
        }
        
        # Sort by priority (highest first)
        sorted_datasets = sorted(selected_datasets.items(), 
                               key=lambda x: x[1]['priority'])
        
        total_datasets = len(sorted_datasets)
        current_idx = 0
        
        for dataset_id, config in sorted_datasets:
            current_idx += 1
            name = config["name"]
            
            logger.info(f"\n📦 [{current_idx}/{total_datasets}] Processing {name}")
            
            # Check if should skip
            dataset_cache_info = cache_status["datasets"].get(dataset_id, {})
            is_cached = dataset_cache_info.get("is_cached", False)
            
            if skip_cached and is_cached and not force_reload:
                cache_size = dataset_cache_info.get("cache_size_mb", 0)
                logger.info(f"    ⏭️  Skipping {name} (already cached, {cache_size:.1f}MB)")
                results["skipped"].append(name)
                continue
            
            # Pre-load or verify the dataset
            success = self.preload_dataset(dataset_id, config, force_reload, verify_only)
            
            if success:
                results["success"].append(name)
            else:
                results["failed"].append(name)
            
            # Brief pause between datasets to avoid overwhelming the system
            if not verify_only:
                time.sleep(1)
        
        # Final summary
        logger.info("\n" + "="*60)
        logger.info(f"📊 {action.title()} Summary:")
        logger.info(f"   ✅ Successful: {len(results['success'])} datasets")
        logger.info(f"   ❌ Failed: {len(results['failed'])} datasets") 
        logger.info(f"   ⏭️  Skipped: {len(results['skipped'])} datasets")
        
        if results["success"]:
            logger.info("   Successful datasets:")
            for name in results["success"]:
                logger.info(f"     • {name}")
        
        if results["failed"]:
            logger.warning("   Failed datasets:")
            for name in results["failed"]:
                logger.warning(f"     • {name}")
        
        if results["skipped"]:
            logger.info("   Skipped datasets:")
            for name in results["skipped"]:
                logger.info(f"     • {name}")
        
        return results
    
    def clear_cache(self, dataset_names: List[str] = None, force: bool = False):
        """FIXED: Clear cache with better safety and specificity"""
        
        if dataset_names is None:
            if not force:
                logger.error("🚨 Refusing to clear ALL dataset cache without --force flag!")
                logger.info("   Use --force if you really want to delete everything")
                logger.info("   Or specify specific datasets with --datasets")
                return False
            
            logger.info("🗑️  Clearing ALL dataset cache...")
            if self.cache_dir.exists():
                import shutil
                try:
                    shutil.rmtree(self.cache_dir)
                    logger.info("   ✅ All cache cleared")
                    return True
                except Exception as e:
                    logger.error(f"   ❌ Failed to clear cache: {e}")
                    return False
            else:
                logger.info("   ℹ️  No cache directory found")
                return True
        else:
            logger.info(f"🗑️  Clearing cache for specific datasets: {dataset_names}")
            
            cleared_count = 0
            cache_status = self.check_cache_status()
            
            for dataset_name in dataset_names:
                # Find dataset by name or ID
                dataset_id = None
                for did, dconfig in self.splatflow_datasets.items():
                    if dconfig["name"] == dataset_name or did == dataset_name:
                        dataset_id = did
                        break
                
                if not dataset_id:
                    logger.warning(f"   ⚠️  Dataset not found: {dataset_name}")
                    continue
                
                # Get cache files for this dataset
                is_cached, cache_files = self._find_dataset_cache_files(
                    dataset_id, self.splatflow_datasets[dataset_id].get("config")
                )
                
                if not is_cached:
                    logger.info(f"   ℹ️  {dataset_name} not cached, skipping")
                    continue
                
                # Remove cache files
                removed_size_mb = 0
                for cache_file in cache_files:
                    try:
                        if cache_file.exists():
                            size_mb = cache_file.stat().st_size / (1024 * 1024)
                            cache_file.parent.rmdir() if cache_file.parent != self.cache_dir else None
                            removed_size_mb += size_mb
                    except Exception as e:
                        logger.warning(f"   ⚠️  Failed to remove {cache_file}: {e}")
                
                if removed_size_mb > 0:
                    logger.info(f"   ✅ Cleared {dataset_name} cache ({removed_size_mb:.1f}MB)")
                    cleared_count += 1
                else:
                    logger.warning(f"   ⚠️  No cache files removed for {dataset_name}")
            
            logger.info(f"   📊 Cleared cache for {cleared_count} datasets")
            return cleared_count > 0
    
    def get_dataset_info(self, dataset_name: str = None) -> Dict:
        """Get detailed information about a specific dataset or all datasets"""
        
        if dataset_name:
            # Find specific dataset
            dataset_id = None
            for did, dconfig in self.splatflow_datasets.items():
                if dconfig["name"] == dataset_name or did == dataset_name:
                    dataset_id = did
                    break
            
            if not dataset_id:
                return {"error": f"Dataset not found: {dataset_name}"}
            
            config = self.splatflow_datasets[dataset_id]
            is_cached, cache_files = self._find_dataset_cache_files(dataset_id, config.get("config"))
            
            return {
                "dataset_id": dataset_id,
                "name": config["name"],
                "description": config["description"],
                "is_cached": is_cached,
                "cache_files_count": len(cache_files),
                "config": config
            }
        else:
            # Return all datasets info
            all_info = {}
            for dataset_id, config in self.splatflow_datasets.items():
                is_cached, cache_files = self._find_dataset_cache_files(dataset_id, config.get("config"))
                all_info[dataset_id] = {
                    "name": config["name"],
                    "description": config["description"],
                    "is_cached": is_cached,
                    "cache_files_count": len(cache_files),
                    "priority": config["priority"]
                }
            
            return all_info


def main():
    """FIXED main function with better argument handling"""
    parser = argparse.ArgumentParser(description="FIXED SplatFlow Dataset Cache Manager")
    
    parser.add_argument("--action", "-a",
                       choices=["status", "download", "verify", "clear", "info"],
                       default="status",
                       help="Action to perform (default: status)")
    
    parser.add_argument("--datasets", "-d", nargs="+",
                       help="Specific datasets to process (by name or ID)")
    
    parser.add_argument("--skip-cached", action="store_true", default=True,
                       help="Skip datasets that are already cached (default: True)")
    
    parser.add_argument("--force", action="store_true",
                       help="Force action even if risky (e.g., clear all cache)")
    
    parser.add_argument("--detailed", action="store_true",
                       help="Show detailed information")
    
    parser.add_argument("--all", action="store_true",
                       help="Process all datasets (override --datasets)")
    
    args = parser.parse_args()
    
    print("🚀 FIXED SplatFlow Dataset Cache Manager")
    print("=" * 60)
    
    manager = FixedSplatFlowDatasetManager()
    
    if args.action == "status":
        print("🔍 Checking dataset cache status...")
        
        cache_status = manager.check_cache_status(detailed=args.detailed)
        summary = cache_status["summary"]
        
        print(f"\n📊 Cache Summary:")
        print(f"   Cached: {summary['cached_count']}/{summary['total_count']} datasets ({summary['cache_percentage']:.1f}%)")
        print(f"   Total size: {summary['total_size_mb']:.1f}MB ({summary['total_size_gb']:.2f}GB)")
        
        if summary['cached_count'] == summary['total_count']:
            print("\n🎉 All SplatFlow datasets are cached!")
            print("   Your next training run will be fast.")
        elif summary['cached_count'] > 0:
            print(f"\n🔄 {summary['cached_count']}/{summary['total_count']} datasets cached.")
            print("   Run with --action download to cache remaining datasets.")
        else:
            print("\n📥 No datasets cached.")
            print("   Run with --action download to cache datasets.")
            total_gb, _ = manager.estimate_download_size()
            print(f"   Estimated download size: {total_gb:.1f}GB")
    
    elif args.action == "download":
        print("📥 Downloading/caching SplatFlow datasets...")
        
        datasets_to_process = None
        if not args.all and args.datasets:
            datasets_to_process = args.datasets
            print(f"   Selected datasets: {datasets_to_process}")
        
        results = manager.preload_selected_datasets(
            dataset_names=datasets_to_process,
            force_reload=args.force,
            skip_cached=args.skip_cached,
            verify_only=False
        )
        
        if len(results["failed"]) == 0:
            print("\n🎉 All selected datasets successfully cached!")
            print("   Your SplatFlow training will now be much faster.")
        else:
            print(f"\n⚠️  {len(results['failed'])} datasets failed to cache.")
            print("   Training will still work but may download during training.")
    
    elif args.action == "verify":
        print("🔍 Verifying cached datasets...")
        
        datasets_to_process = None
        if not args.all and args.datasets:
            datasets_to_process = args.datasets
        
        results = manager.preload_selected_datasets(
            dataset_names=datasets_to_process,
            force_reload=False,
            skip_cached=False,  # Verify all, even cached ones
            verify_only=True
        )
        
        print(f"\n📊 Verification complete:")
        print(f"   ✅ Verified: {len(results['success'])} datasets")
        print(f"   ❌ Failed: {len(results['failed'])} datasets")
    
    elif args.action == "clear":
        print("🗑️  Clearing dataset cache...")
        
        if args.datasets:
            success = manager.clear_cache(dataset_names=args.datasets, force=args.force)
        else:
            if not args.force:
                print("⚠️  This will delete ALL cached datasets!")
                confirm = input("Are you sure? Type 'yes' to confirm: ")
                if confirm.lower() != 'yes':
                    print("❌ Cache clear cancelled.")
                    return
            
            success = manager.clear_cache(force=True)
        
        if success:
            print("✅ Cache cleared successfully.")
        else:
            print("❌ Cache clear failed or was cancelled.")
    
    elif args.action == "info":
        print("ℹ️  Dataset information...")
        
        if args.datasets and len(args.datasets) == 1:
            info = manager.get_dataset_info(args.datasets[0])
            if "error" in info:
                print(f"❌ {info['error']}")
            else:
                print(f"\n📊 Dataset: {info['name']}")
                print(f"   ID: {info['dataset_id']}")
                print(f"   Description: {info['description']}")
                print(f"   Cached: {'✅' if info['is_cached'] else '❌'}")
                print(f"   Cache files: {info['cache_files_count']}")
        else:
            all_info = manager.get_dataset_info()
            print(f"\n📊 All SplatFlow Datasets:")
            for dataset_id, info in sorted(all_info.items(), key=lambda x: x[1]['priority']):
                status = "✅" if info['is_cached'] else "❌"
                print(f"   {info['priority']}. {info['name']}: {status}")
                print(f"      {info['description']}")
                if args.detailed:
                    print(f"      ID: {dataset_id}")
                    print(f"      Cache files: {info['cache_files_count']}")
    
    print("\n" + "=" * 60)
    print("💡 Usage Tips:")
    print("   • python cache.py --action status          # Check what's cached")
    print("   • python cache.py --action download        # Download all datasets")
    print("   • python cache.py --action download -d TinyStories  # Download specific dataset")
    print("   • python cache.py --action verify          # Verify cached datasets work")
    print("   • python cache.py --action clear -d TinyStories    # Clear specific dataset")
    print("   • python cache.py --action info            # Show all dataset info")


if __name__ == "__main__":
    main()
